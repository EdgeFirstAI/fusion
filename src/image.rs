use async_pidfd::PidFd;
use core::fmt;
use dma_buf::DmaBuf;
use dma_heap::{Heap, HeapKind};
use edgefirst_schemas::edgefirst_msgs::DmaBuf as DmaBufMsg;
use g2d_sys::{
    fourcc::FourCC, g2d as g2d_library, g2d_buf, g2d_rotation_G2D_ROTATION_0,
    g2d_rotation_G2D_ROTATION_180, g2d_rotation_G2D_ROTATION_270, g2d_rotation_G2D_ROTATION_90,
    g2d_surface, g2d_surface_new, guess_version, G2DFormat, G2DPhysical,
};
use libc::{dup, mmap, munmap, MAP_SHARED, PROT_READ, PROT_WRITE};
use log::{debug, warn};
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use std::{
    error::Error,
    ffi::c_void,
    io,
    os::{
        fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd},
        unix::io::OwnedFd,
    },
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
};

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

#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
pub enum Rotation {
    Rotation0 = g2d_rotation_G2D_ROTATION_0 as isize,
    Rotation90 = g2d_rotation_G2D_ROTATION_90 as isize,
    Rotation180 = g2d_rotation_G2D_ROTATION_180 as isize,
    Rotation270 = g2d_rotation_G2D_ROTATION_270 as isize,
}
pub struct G2DBuffer<'a> {
    buf: *mut g2d_buf,
    imgmgr: &'a ImageManager,
}

#[allow(dead_code)]
impl G2DBuffer<'_> {
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

impl Drop for G2DBuffer<'_> {
    fn drop(&mut self) {
        self.imgmgr.free(self);
        debug!("G2D Buffer freed")
    }
}

pub struct ImageManager {
    lib: g2d_library,
    version: g2d_sys::Version,
    handle: *mut c_void,
}

const G2D_2_3_0: g2d_sys::Version = g2d_sys::Version {
    major: 6,
    minor: 4,
    patch: 11,
    num: 1049711,
};

impl ImageManager {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let lib = unsafe { g2d_library::new("libg2d.so.2") }?;
        let mut handle: *mut c_void = null_mut();

        if unsafe { lib.g2d_open(&mut handle) } != 0 {
            let err = io::Error::last_os_error();
            return Err(Box::new(err));
        }
        let version = guess_version(&lib).unwrap_or(G2D_2_3_0);
        Ok(Self {
            lib,
            handle,
            version,
        })
    }

    pub fn version(&self) -> g2d_sys::Version {
        self.version
    }

    #[allow(dead_code)]
    pub fn alloc(
        &self,
        width: i32,
        height: i32,
        channels: i32,
    ) -> Result<G2DBuffer, Box<dyn Error>> {
        let g2d_buf = unsafe { self.lib.g2d_alloc(width * height * channels, 0) };
        if g2d_buf.is_null() {
            return Err(Box::new(io::Error::other("g2d_alloc failed")));
        }
        debug!("G2D Buffer alloc'd");
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

    #[allow(dead_code)]
    pub fn g2d_buf_fd(&self, buf: &G2DBuffer) -> OwnedFd {
        let fd = unsafe { self.lib.g2d_buf_export_fd(buf.buf) };
        unsafe { OwnedFd::from_raw_fd(fd) }
    }

    pub fn convert(
        &self,
        from: &Image,
        to: &Image,
        crop: Option<Rect>,
        rot: Rotation,
    ) -> Result<(), Box<dyn Error>> {
        if self.version >= G2D_2_3_0 {
            self.convert_new(from, to, crop, rot)
        } else {
            self.convert_old(from, to, crop, rot)
        }
    }

    pub fn convert_old(
        &self,
        from: &Image,
        to: &Image,
        crop: Option<Rect>,
        rot: Rotation,
    ) -> Result<(), Box<dyn Error>> {
        let mut src: g2d_surface = from.try_into()?;

        if let Some(r) = crop {
            src.left = r.x;
            src.top = r.y;
            src.right = r.x + r.width;
            src.bottom = r.y + r.height;
        }

        let mut dst: g2d_surface = to.try_into()?;

        dst.rot = rot as u32;

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

    pub fn convert_new(
        &self,
        from: &Image,
        to: &Image,
        crop: Option<Rect>,
        rot: Rotation,
    ) -> Result<(), Box<dyn Error>> {
        let mut src: g2d_surface_new = from.try_into()?;

        if let Some(r) = crop {
            src.left = r.x;
            src.top = r.y;
            src.right = r.x + r.width;
            src.bottom = r.y + r.height;
        }

        let mut dst: g2d_surface_new = to.try_into()?;

        dst.rot = rot as u32;

        if unsafe {
            // force cast the g2d_surface_new to g2d_surface so it can be sent to the
            // g2d_blit function
            self.lib.g2d_blit(
                self.handle,
                &raw mut src as *mut g2d_surface,
                &raw mut dst as *mut g2d_surface,
            )
        } != 0
        {
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
        debug!("G2D closed");
    }
}

#[derive(Debug)]
pub struct Image {
    pub fd: OwnedFd,
    pub width: u32,
    pub height: u32,
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

    pub fn mmap(&mut self) -> MappedImage {
        let image_size = image_size(self.width, self.height, self.format);
        unsafe {
            let mmap = mmap(
                null_mut(),
                image_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                self.raw_fd(),
                0,
            ) as *mut u8;
            MappedImage {
                mmap,
                len: image_size,
            }
        }
    }
}

impl TryFrom<&Image> for g2d_surface {
    type Error = std::io::Error;

    fn try_from(img: &Image) -> Result<Self, Self::Error> {
        let to_fd = img.fd.try_clone()?;
        let to_phys: G2DPhysical = DmaBuf::from(to_fd).into();
        Ok(Self {
            planes: [to_phys.into(), 0, 0],
            format: G2DFormat::from(img.format).format(),
            left: 0,
            top: 0,
            right: img.width as i32,
            bottom: img.height as i32,
            stride: img.width as i32,
            width: img.width as i32,
            height: img.height as i32,
            blendfunc: 0,
            clrcolor: 0,
            rot: 0,
            global_alpha: 0,
        })
    }
}

impl TryFrom<&Image> for g2d_surface_new {
    type Error = std::io::Error;

    fn try_from(img: &Image) -> Result<Self, Self::Error> {
        let to_fd = img.fd.try_clone()?;
        let to_phys: G2DPhysical = DmaBuf::from(to_fd).into();
        Ok(Self {
            planes: [to_phys.into(), 0, 0],
            format: G2DFormat::from(img.format).format(),
            left: 0,
            top: 0,
            right: img.width as i32,
            bottom: img.height as i32,
            stride: img.width as i32,
            width: img.width as i32,
            height: img.height as i32,
            blendfunc: 0,
            clrcolor: 0,
            rot: 0,
            global_alpha: 0,
        })
    }
}

impl TryFrom<&DmaBufMsg> for Image {
    type Error = io::Error;

    fn try_from(dma_buf: &DmaBufMsg) -> Result<Self, io::Error> {
        let pidfd: PidFd = PidFd::from_pid(dma_buf.pid as i32)?;
        let fd = get_file_from_pidfd(pidfd.as_raw_fd(), dma_buf.fd, GetFdFlags::empty())?;
        let fourcc = dma_buf.fourcc.into();
        // println!("src fourcc: {:?}", fourcc);
        Ok(Image {
            fd: fd.into(),
            width: dma_buf.width,
            height: dma_buf.height,
            format: fourcc,
        })
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

pub struct MappedImage {
    mmap: *mut u8,
    len: usize,
}

impl MappedImage {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { from_raw_parts(self.mmap, self.len) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { from_raw_parts_mut(self.mmap, self.len) }
    }
}
impl Drop for MappedImage {
    fn drop(&mut self) {
        if unsafe { munmap(self.mmap.cast::<c_void>(), self.len) } > 0 {
            warn!("unmap failed!");
        }
    }
}
