// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use async_pidfd::PidFd;
use core::fmt;
use dma_buf::DmaBuf;
use dma_heap::{Heap, HeapKind};
use edgefirst_schemas::edgefirst_msgs::DmaBuffer as DmaBufMsg;
use four_char_code::{four_char_code, FourCharCode};
use g2d_sys::{
    g2d_rotation_G2D_ROTATION_0, g2d_rotation_G2D_ROTATION_180, g2d_rotation_G2D_ROTATION_270,
    g2d_rotation_G2D_ROTATION_90, G2DFormat, G2DPhysical, G2DSurface, G2D,
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

/// Local type alias for readability across the codebase.
pub type FourCC = FourCharCode;

pub const RGB3: FourCC = four_char_code!("RGB3");
pub const RGBX: FourCC = four_char_code!("RGBX");
pub const RGBA: FourCC = four_char_code!("RGBA");
pub const YUYV: FourCC = four_char_code!("YUYV");
pub const NV12: FourCC = four_char_code!("NV12");

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

pub struct ImageManager {
    g2d: G2D,
}

impl ImageManager {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let g2d = G2D::new("libg2d.so.2")?;
        debug!("G2D version: {}", g2d.version());
        Ok(Self { g2d })
    }

    pub fn version(&self) -> g2d_sys::Version {
        self.g2d.version()
    }

    pub fn convert(
        &self,
        from: &Image,
        to: &Image,
        crop: Option<Rect>,
        rot: Rotation,
    ) -> Result<(), Box<dyn Error>> {
        let mut src: G2DSurface = from.try_into()?;

        if let Some(r) = crop {
            src.left = r.x;
            src.top = r.y;
            src.right = r.x + r.width;
            src.bottom = r.y + r.height;
        }

        let mut dst: G2DSurface = to.try_into()?;
        dst.rot = rot as u32;

        self.g2d.blit(&src, &dst)?;
        self.g2d.finish()?;
        // TODO(hardware): G2D output buffer may require cache invalidation
        // on i.MX8M Plus when DMA coherency is not guaranteed.

        Ok(())
    }
}

pub struct Image {
    pub fd: OwnedFd,
    pub width: u32,
    pub height: u32,
    pub format: FourCC,
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("fd", &self.fd)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &fourcc_str(self.format))
            .finish()
    }
}

/// Returns the average bytes per row for the given format, used to calculate
/// total image buffer size. Note: for planar formats like NV12, this is NOT
/// the actual row stride but rather total_size/height.
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
        let ptr = unsafe {
            mmap(
                null_mut(),
                image_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                self.raw_fd(),
                0,
            )
        };
        assert!(ptr != libc::MAP_FAILED, "mmap failed");
        MappedImage {
            mmap: ptr as *mut u8,
            len: image_size,
        }
    }
}

impl TryFrom<&Image> for G2DSurface {
    type Error = Box<dyn Error>;

    fn try_from(img: &Image) -> Result<Self, Self::Error> {
        let phys = G2DPhysical::try_from(img.fd.as_raw_fd())?;
        let format = G2DFormat::try_from(img.format)?.format();
        Ok(Self {
            planes: [phys.address(), 0, 0],
            format,
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
        let fourcc = FourCharCode::new(dma_buf.fourcc)
            .map_err(|e| io::Error::other(format!("invalid fourcc: {e}")))?;
        Ok(Image {
            fd: fd.into(),
            width: dma_buf.width,
            height: dma_buf.height,
            format: fourcc,
        })
    }
}

/// Format a FourCharCode as a 4-character string for display purposes.
fn fourcc_str(fcc: FourCC) -> String {
    // FourCharCode stores a u32; extract the 4 ASCII bytes
    let val: u32 = fcc.into();
    let bytes = val.to_le_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}x{} {} fd:{:?}",
            self.width,
            self.height,
            fourcc_str(self.format),
            self.fd
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
        if unsafe { munmap(self.mmap.cast::<c_void>(), self.len) } != 0 {
            warn!("unmap failed!");
        }
    }
}
