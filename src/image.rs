use core::fmt;
use dma_buf::DmaBuf;
use dma_heap::{Heap, HeapKind};
use g2d_sys::{
    g2d as g2d_library, g2d_buf, g2d_rotation_G2D_ROTATION_0, g2d_rotation_G2D_ROTATION_180,
    g2d_rotation_G2D_ROTATION_270, g2d_rotation_G2D_ROTATION_90, g2d_surface, G2DFormat,
    G2DPhysical,
};
use libc::dup;
use std::{
    error::Error,
    ffi::c_void,
    io,
    os::{
        fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd},
        unix::io::OwnedFd,
    },
    ptr::null_mut,
};
use videostream::{camera::CameraBuffer, encoder::VSLRect, fourcc::FourCC, frame::Frame};

pub const RGB3: FourCC = FourCC(*b"RGB3");
pub const RGBX: FourCC = FourCC(*b"RGBX");
pub const RGBA: FourCC = FourCC(*b"RGBA");
pub const YUYV: FourCC = FourCC(*b"YUYV");
pub const NV12: FourCC = FourCC(*b"NV12");

pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl From<VSLRect> for Rect {
    fn from(value: VSLRect) -> Self {
        Rect {
            x: value.get_x(),
            y: value.get_y(),
            width: value.get_width(),
            height: value.get_height(),
        }
    }
}

pub struct G2DBuffer<'a> {
    buf: *mut g2d_buf,
    imgmgr: &'a ImageManager,
}

#[derive(Copy, Clone, Debug)]
pub enum Rotation {
    Rotation0 = g2d_rotation_G2D_ROTATION_0 as isize,
    Rotation90 = g2d_rotation_G2D_ROTATION_90 as isize,
    Rotation180 = g2d_rotation_G2D_ROTATION_180 as isize,
    Rotation270 = g2d_rotation_G2D_ROTATION_270 as isize,
}

impl<'a> G2DBuffer<'a> {
    pub unsafe fn buf_handle(&self) -> *mut c_void {
        (*self.buf).buf_handle
    }

    pub unsafe fn buf_vaddr(&self) -> *mut c_void {
        (*self.buf).buf_vaddr
    }

    pub fn buf_paddr(&self) -> i32 {
        unsafe { (*self.buf).buf_paddr }
    }

    pub fn buf_size(&self) -> i32 {
        unsafe { (*self.buf).buf_size }
    }
}

impl<'a> Drop for G2DBuffer<'a> {
    fn drop(&mut self) {
        self.imgmgr.free(self);
    }
}

pub struct ImageManager {
    lib: g2d_library,
    handle: *mut c_void,
}

impl ImageManager {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let lib = unsafe { g2d_library::new("libg2d.so.2") }?;
        let mut handle: *mut c_void = null_mut();

        if unsafe { lib.g2d_open(&mut handle) } != 0 {
            let err = io::Error::last_os_error();
            return Err(Box::new(err));
        }
        Ok(Self { lib, handle })
    }

    pub fn alloc(
        &self,
        width: usize,
        height: usize,
        channels: usize,
    ) -> Result<G2DBuffer, Box<dyn Error>> {
        let g2d_buf = unsafe { self.lib.g2d_alloc((width * height * channels) as i32, 0) };
        if g2d_buf.is_null() {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::Other,
                "g2d_alloc failed",
            )));
        }
        Ok(G2DBuffer {
            buf: g2d_buf,
            imgmgr: self,
        })
    }

    pub fn free(&self, buf: &mut G2DBuffer) {
        unsafe {
            self.lib.g2d_free(buf.buf);
        }
    }

    pub fn g2d_buf_fd(&self, buf: &G2DBuffer) -> OwnedFd {
        let fd = unsafe { self.lib.g2d_buf_export_fd(buf.buf) };
        unsafe { OwnedFd::from_raw_fd(fd) }
    }

    pub fn convert_phys(
        &self,
        from: &Frame,
        to: &Image,
        crop: &Option<Rect>,
        rot: Rotation,
    ) -> Result<(), Box<dyn Error>> {
        let from_phys: G2DPhysical = match from.paddr() {
            Some(v) => (v as i32).into(),
            None => unsafe { DmaBuf::from_raw_fd(from.handle()).into() },
        };
        let to_fd = to.fd.try_clone()?;
        let to_phys: G2DPhysical = DmaBuf::from(to_fd).into();
        let fourcc = FourCC::from(from.fourcc());
        let planes = match fourcc {
            NV12 => {
                let width = from.width();
                let height = from.height();
                let y_size = width * height;
                let v_size = y_size / 4;
                let phys = from_phys.into();
                [phys, phys + y_size, phys + y_size + v_size]
            }
            _ => [from_phys.into(), 0, 0],
        };
        let mut src = g2d_surface {
            planes,
            format: G2DFormat::from(fourcc).format(),
            left: 0,
            top: 0,
            right: from.width(),
            bottom: from.height(),
            stride: from.width(),
            width: from.width(),
            height: from.height(),
            blendfunc: 0,
            clrcolor: 0,
            rot: 0,
            global_alpha: 0,
        };

        if let Some(r) = crop {
            src.left = r.x;
            src.top = r.y;
            src.right = r.x + r.width;
            src.bottom = r.y + r.height;
        }

        let mut dst = g2d_surface {
            planes: [to_phys.into(), 0, 0],
            format: G2DFormat::from(to.format).format(),
            left: 0,
            top: 0,
            right: to.width as i32,
            bottom: to.height as i32,
            stride: to.width as i32,
            width: to.width as i32,
            height: to.height as i32,
            blendfunc: 0,
            clrcolor: 0,
            rot: rot as u32,
            global_alpha: 0,
        };

        if unsafe { self.lib.g2d_blit(self.handle, &mut src, &mut dst) } != 0 {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidInput,
                "g2d_blit failed",
            )));
        }
        if unsafe { self.lib.g2d_finish(self.handle) } != 0 {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidInput,
                "g2d_finish failed",
            )));
        }
        // FIXME: A cache invalidation is required here, currently missing!

        Ok(())
    }

    pub fn convert(
        &self,
        from: &Image,
        to: &Image,
        crop: Option<Rect>,
    ) -> Result<(), Box<dyn Error>> {
        let from_fd = from.fd.try_clone()?;
        let from_phys: G2DPhysical = DmaBuf::from(from_fd).into();

        let to_fd = to.fd.try_clone()?;
        let to_phys: G2DPhysical = DmaBuf::from(to_fd).into();

        let mut src = g2d_surface {
            planes: [from_phys.into(), 0, 0],
            format: G2DFormat::from(from.format).format(),
            left: 0,
            top: 0,
            right: from.width as i32,
            bottom: from.height as i32,
            stride: from.width as i32,
            width: from.width as i32,
            height: from.height as i32,
            blendfunc: 0,
            clrcolor: 0,
            rot: 0,
            global_alpha: 0,
        };

        if let Some(r) = crop {
            src.left = r.x;
            src.top = r.y;
            src.right = r.x + r.width;
            src.bottom = r.y + r.height;
        }

        let mut dst = g2d_surface {
            planes: [to_phys.into(), 0, 0],
            format: G2DFormat::from(to.format).format(),
            left: 0,
            top: 0,
            right: to.width as i32,
            bottom: to.height as i32,
            stride: to.width as i32,
            width: to.width as i32,
            height: to.height as i32,
            blendfunc: 0,
            clrcolor: 0,
            rot: 0,
            global_alpha: 0,
        };

        if unsafe { self.lib.g2d_blit(self.handle, &mut src, &mut dst) } != 0 {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidInput,
                "g2d_blit failed",
            )));
        }
        if unsafe { self.lib.g2d_finish(self.handle) } != 0 {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidInput,
                "g2d_finish failed",
            )));
        }

        // FIXME: A cache invalidation is required here, currently missing!

        Ok(())
    }
}

impl Drop for ImageManager {
    fn drop(&mut self) {
        _ = unsafe { self.lib.g2d_close(self.handle) };
    }
}

#[derive(Debug)]
pub struct Image {
    fd: OwnedFd,
    width: u32,
    height: u32,
    pub format: FourCC,
}

const fn format_row_stride(format: FourCC, width: u32) -> usize {
    match format {
        RGB3 => 3 * width as usize,
        RGBX => 4 * width as usize,
        RGBA => 4 * width as usize,
        YUYV => 2 * width as usize,
        NV12 => width as usize / 2 + width as usize,
        _ => todo!(),
    }
}

const fn image_size(width: u32, height: u32, format: FourCC) -> usize {
    format_row_stride(format, width) * height as usize
}

impl Image {
    pub fn new(width: u32, height: u32, format: FourCC) -> Result<Self, Box<dyn Error>> {
        let heap = Heap::new(HeapKind::Cma)?;
        let fd = heap.allocate(image_size(width, height, format))?;
        Ok(Self {
            fd,
            width,
            height,
            format,
        })
    }

    pub fn new_preallocated(fd: OwnedFd, width: u32, height: u32, format: FourCC) -> Self {
        Self {
            fd,
            width,
            height,
            format,
        }
    }

    pub fn from_camera(buffer: &CameraBuffer) -> Result<Self, Box<dyn Error>> {
        let fd = buffer.fd();

        Ok(Self {
            fd: fd.try_clone_to_owned()?,
            width: buffer.width() as u32,
            height: buffer.height() as u32,
            format: buffer.format(),
        })
    }

    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    pub fn raw_fd(&self) -> i32 {
        self.fd.as_raw_fd()
    }

    pub fn dmabuf(&self) -> DmaBuf {
        unsafe { DmaBuf::from_raw_fd(dup(self.fd.as_raw_fd())) }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> FourCC {
        self.format
    }

    pub fn size(&self) -> usize {
        format_row_stride(self.format, self.width) * self.height as usize
    }
}

impl TryFrom<&Image> for Frame {
    type Error = Box<dyn Error>;

    fn try_from(img: &Image) -> Result<Self, Self::Error> {
        let frame = match Frame::new(
            img.width(),
            img.height(),
            0,
            img.format().to_string().as_str(),
        ) {
            Ok(f) => f,
            Err(e) => return Err(e),
        };
        match frame.attach(img.fd().as_raw_fd(), 0, 0) {
            Ok(_) => (),
            Err(e) => return Err(e),
        }
        Ok(frame)
    }
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}x{} {} fd:{:?}",
            self.width, self.height, self.format, self.fd
        )
    }
}
