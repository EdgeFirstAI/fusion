// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use async_pidfd::PidFd;
use core::fmt;
use dma_heap::{Heap, HeapKind};
use edgefirst_schemas::edgefirst_msgs::CameraFrame;
use g2d_sys::{
    g2d_format, g2d_format_G2D_NV12, g2d_format_G2D_RGB888, g2d_format_G2D_RGBA8888,
    g2d_format_G2D_RGBX8888, g2d_format_G2D_YUYV, g2d_rotation_G2D_ROTATION_0,
    g2d_rotation_G2D_ROTATION_180, g2d_rotation_G2D_ROTATION_270, g2d_rotation_G2D_ROTATION_90,
    G2DPhysical, G2DSurface, G2D,
};
use libc::{mmap, munmap, MAP_SHARED, PROT_READ, PROT_WRITE};
use log::{debug, warn};
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use std::{
    error::Error,
    ffi::c_void,
    io,
    os::{fd::AsRawFd, unix::io::OwnedFd},
    ptr::null_mut,
    slice::from_raw_parts_mut,
};

/// HAL / V4L2 fourcc stored as four ASCII bytes in wire order (`b"YUYV"`).
/// Matches the camera service (`videostream::fourcc::FourCC(*b"YUYV")`) and
/// the CameraFrame tensor `format` string. Do not route these through
/// `four-char-code` / `G2DFormat::try_from` — that path byte-swaps to VYUY.
pub type FourCC = [u8; 4];

pub const RGB3: FourCC = *b"RGB3";
pub const RGBX: FourCC = *b"RGBX";
pub const RGBA: FourCC = *b"RGBA";
pub const YUYV: FourCC = *b"YUYV";
pub const NV12: FourCC = *b"NV12";

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
        let mut src = surface_from_image(from)?;

        if let Some(r) = crop {
            src.left = r.x;
            src.top = r.y;
            src.right = r.x + r.width;
            src.bottom = r.y + r.height;
        }

        let mut dst = surface_from_image(to)?;
        dst.rot = rot as u32;

        self.g2d.blit(&src, &dst)?;
        self.g2d.finish()?;
        // TODO(hardware): G2D output buffer may require cache invalidation
        // on i.MX8M Plus when DMA coherency is not guaranteed.

        Ok(())
    }
}

/// Map a HAL fourcc (same table as `edgefirst-camera::image::fourcc_to_g2d_format`)
/// to the G2D format constant. YUYV must not become VYUY; RGBA must not become ABGR.
pub fn fourcc_to_g2d_format(fourcc: FourCC) -> Result<g2d_format, io::Error> {
    match &fourcc {
        b"RGB3" => Ok(g2d_format_G2D_RGB888),
        b"RGBX" => Ok(g2d_format_G2D_RGBX8888),
        b"RGBA" => Ok(g2d_format_G2D_RGBA8888),
        b"YUYV" => Ok(g2d_format_G2D_YUYV),
        b"NV12" => Ok(g2d_format_G2D_NV12),
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("unsupported G2D pixel format: {}", fourcc_str(fourcc)),
        )),
    }
}

/// Parse the CameraFrame / HAL tensor `format` string as four ASCII bytes.
pub fn fourcc_from_hal(format: &str) -> io::Result<FourCC> {
    let bytes = format.as_bytes();
    if bytes.len() != 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("HAL fourcc must be 4 characters, got {format:?}"),
        ));
    }
    Ok([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Build a [`G2DSurface`] from an [`Image`]'s DMA buffer (camera `surface_from_image`).
fn surface_from_image(img: &Image) -> Result<G2DSurface, Box<dyn Error>> {
    let phys = G2DPhysical::new(img.fd.as_raw_fd())?;
    let addr = phys.address();
    let planes = match img.format {
        NV12 => {
            let y_size = img.width as u64 * img.height as u64;
            [addr, addr + y_size, 0]
        }
        _ => [addr, 0, 0],
    };
    Ok(G2DSurface {
        planes,
        format: fourcc_to_g2d_format(img.format)?,
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

    pub fn raw_fd(&self) -> i32 {
        self.fd.as_raw_fd()
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

fn camera_frame_invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn camera_frame_u32(name: &str, value: u64) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| {
        camera_frame_invalid(format!(
            "CameraFrame tensor {name} {value} does not fit in u32"
        ))
    })
}

fn camera_frame_nonzero_u32(name: &str, value: u64) -> io::Result<u32> {
    let dim = camera_frame_u32(name, value)?;
    if dim == 0 {
        return Err(camera_frame_invalid(format!(
            "CameraFrame tensor {name} is 0"
        )));
    }
    Ok(dim)
}

/// Import a [`CameraFrame`] tensor plane 0 as an [`Image`] via pidfd + getfd.
pub fn image_from_camera_frame(frame: &CameraFrame<Vec<u8>>) -> Result<Image, io::Error> {
    let t = frame.tensor();
    let plane = t
        .plane_at(0)
        .ok_or_else(|| camera_frame_invalid("CameraFrame tensor has no plane 0"))?;
    let pid = t.pid();
    let pid_i32 = i32::try_from(pid)
        .map_err(|_| camera_frame_invalid(format!("CameraFrame tensor pid {pid} exceeds i32")))?;
    let pidfd: PidFd = PidFd::from_pid(pid_i32)?;
    let target_fd = i32::try_from(plane.handle).map_err(|_| {
        camera_frame_invalid(format!(
            "CameraFrame plane handle {} exceeds i32",
            plane.handle
        ))
    })?;
    let fd = get_file_from_pidfd(pidfd.as_raw_fd(), target_fd, GetFdFlags::empty())?;
    let height = t
        .shape_at(0)
        .ok_or_else(|| camera_frame_invalid("CameraFrame tensor missing height (shape[0])"))?;
    let width = t
        .shape_at(1)
        .ok_or_else(|| camera_frame_invalid("CameraFrame tensor missing width (shape[1])"))?;
    let fourcc = fourcc_from_hal(t.format())?;
    Ok(Image {
        fd: fd.into(),
        width: camera_frame_nonzero_u32("width", width)?,
        height: camera_frame_nonzero_u32("height", height)?,
        format: fourcc,
    })
}

impl TryFrom<&CameraFrame<Vec<u8>>> for Image {
    type Error = io::Error;

    fn try_from(frame: &CameraFrame<Vec<u8>>) -> Result<Self, io::Error> {
        image_from_camera_frame(frame)
    }
}

/// Format a HAL fourcc as a 4-character string for display purposes.
fn fourcc_str(fcc: FourCC) -> String {
    String::from_utf8_lossy(&fcc).into_owned()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourcc_from_hal_keeps_wire_order() {
        assert_eq!(fourcc_from_hal("YUYV").unwrap(), YUYV);
        assert_eq!(fourcc_from_hal("RGBA").unwrap(), RGBA);
        assert_eq!(fourcc_from_hal("NV12").unwrap(), NV12);
        assert!(fourcc_from_hal("YUY").is_err());
        assert!(fourcc_from_hal("ABGRX").is_err());
    }

    #[test]
    fn fourcc_to_g2d_does_not_byte_swap() {
        assert_eq!(fourcc_to_g2d_format(YUYV).unwrap(), g2d_format_G2D_YUYV);
        assert_eq!(fourcc_to_g2d_format(RGBA).unwrap(), g2d_format_G2D_RGBA8888);
        assert_eq!(fourcc_to_g2d_format(RGB3).unwrap(), g2d_format_G2D_RGB888);
        assert_eq!(fourcc_to_g2d_format(RGBX).unwrap(), g2d_format_G2D_RGBX8888);
        assert_eq!(fourcc_to_g2d_format(NV12).unwrap(), g2d_format_G2D_NV12);
        assert!(fourcc_to_g2d_format(*b"VYUY").is_err());
        assert!(fourcc_to_g2d_format(*b"ABGR").is_err());
    }

    #[test]
    fn fourcc_str_matches_hal() {
        assert_eq!(fourcc_str(YUYV), "YUYV");
        assert_eq!(fourcc_str(RGBA), "RGBA");
    }
}
