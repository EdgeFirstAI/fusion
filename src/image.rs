// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use async_pidfd::PidFd;
use core::fmt;
use dma_heap::{Heap, HeapKind};
use edgefirst_schemas::edgefirst_msgs::CameraFrame;
pub use edgefirst_tensor::PixelFormat;
use g2d_sys::{
    g2d_format, g2d_format_G2D_NV12, g2d_format_G2D_RGB888, g2d_format_G2D_RGBA8888,
    g2d_format_G2D_YUYV, g2d_rotation_G2D_ROTATION_0, g2d_rotation_G2D_ROTATION_180,
    g2d_rotation_G2D_ROTATION_270, g2d_rotation_G2D_ROTATION_90, G2DPhysical, G2DSurface, G2D,
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

/// `Tensor.format` strings published by edgefirst-camera 2.x that are not
/// HAL wire names, accepted for compatibility with camera 2.x.
///
/// Camera 2.x publishes the V4L2 fourcc of its capture buffer. `YUYV` and
/// `NV12` already equal the HAL wire names and need no alias. `RGBX` maps to
/// [`PixelFormat::Rgba`]: the byte layout is identical and fusion only reads
/// these frames as an unblended G2D source, so the fourth byte is ignored.
pub const LEGACY_FORMAT_ALIASES: &[(&str, PixelFormat)] = &[
    ("RGB3", PixelFormat::Rgb),
    ("RGBA", PixelFormat::Rgba),
    ("RGBX", PixelFormat::Rgba),
];

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

/// Map a HAL [`PixelFormat`] to the G2D format constant G2D can blit from or to.
pub fn pixel_format_to_g2d_format(format: PixelFormat) -> Result<g2d_format, io::Error> {
    match format {
        PixelFormat::Rgb => Ok(g2d_format_G2D_RGB888),
        PixelFormat::Rgba => Ok(g2d_format_G2D_RGBA8888),
        PixelFormat::Yuyv => Ok(g2d_format_G2D_YUYV),
        PixelFormat::Nv12 => Ok(g2d_format_G2D_NV12),
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("unsupported G2D pixel format: {}", format.as_str()),
        )),
    }
}

/// Parse a CameraFrame `Tensor.format` string.
///
/// Accepts the HAL wire names ([`PixelFormat::as_str`]) and, for
/// compatibility with camera 2.x, the names in [`LEGACY_FORMAT_ALIASES`].
pub fn pixel_format_from_wire(format: &str) -> io::Result<PixelFormat> {
    PixelFormat::from_str_code(format)
        .or_else(|| {
            LEGACY_FORMAT_ALIASES
                .iter()
                .find(|(name, _)| *name == format)
                .map(|&(_, pf)| pf)
        })
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown CameraFrame tensor format {format:?}"),
            )
        })
}

/// Build a [`G2DSurface`] from an [`Image`]'s DMA buffer (camera `surface_from_image`).
fn surface_from_image(img: &Image) -> Result<G2DSurface, Box<dyn Error>> {
    let phys = G2DPhysical::new(img.fd.as_raw_fd())?;
    let addr = phys.address();
    let planes = match img.format {
        PixelFormat::Nv12 => {
            let y_size = img.width as u64 * img.height as u64;
            [addr, addr + y_size, 0]
        }
        _ => [addr, 0, 0],
    };
    Ok(G2DSurface {
        planes,
        format: pixel_format_to_g2d_format(img.format)?,
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
    pub format: PixelFormat,
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("fd", &self.fd)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format.as_str())
            .finish()
    }
}

/// Bytes needed to hold a `width`x`height` image of `format`, from HAL's
/// allocation geometry. Every [`PixelFormat`] has one-byte samples, so the
/// element count of the allocation shape is the byte count. Returns 0 when
/// HAL has no allocation geometry for the format.
fn image_size(width: u32, height: u32, format: PixelFormat) -> usize {
    format
        .allocation_shape(width as usize, height as usize)
        .map_or(0, |shape| shape.iter().product())
}

impl Image {
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Result<Self, Box<dyn Error>> {
        let size = image_size(width, height, format);
        if size == 0 {
            return Err(format!(
                "cannot allocate a {width}x{height} {} image",
                format.as_str()
            )
            .into());
        }
        let heap = Heap::new(HeapKind::Cma)?;
        let fd = heap.allocate(size)?;
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

    pub fn format(&self) -> PixelFormat {
        self.format
    }

    pub fn size(&self) -> usize {
        image_size(self.width, self.height, self.format)
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
    let format = pixel_format_from_wire(t.format())?;
    Ok(Image {
        fd: fd.into(),
        width: camera_frame_nonzero_u32("width", width)?,
        height: camera_frame_nonzero_u32("height", height)?,
        format,
    })
}

impl TryFrom<&CameraFrame<Vec<u8>>> for Image {
    type Error = io::Error;

    fn try_from(frame: &CameraFrame<Vec<u8>>) -> Result<Self, io::Error> {
        image_from_camera_frame(frame)
    }
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}x{} {} fd:{:?}",
            self.width,
            self.height,
            self.format.as_str(),
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

    /// Every HAL wire name parses to its own format.
    #[test]
    fn parses_every_hal_wire_name() {
        for &pf in PixelFormat::all() {
            assert_eq!(pixel_format_from_wire(pf.as_str()).unwrap(), pf);
        }
    }

    /// The formats the camera publishes, by both their HAL and camera 2.x names.
    #[test]
    fn parses_camera_formats_by_hal_and_legacy_name() {
        let cases = [
            ("YUYV", "YUYV", PixelFormat::Yuyv),
            ("NV12", "NV12", PixelFormat::Nv12),
            ("rgb8", "RGB3", PixelFormat::Rgb),
            ("rgba8", "RGBA", PixelFormat::Rgba),
            ("rgba8", "RGBX", PixelFormat::Rgba),
        ];
        for (hal, legacy, expected) in cases {
            assert_eq!(expected.as_str(), hal);
            assert_eq!(pixel_format_from_wire(hal).unwrap(), expected);
            assert_eq!(pixel_format_from_wire(legacy).unwrap(), expected);
        }
    }

    /// The alias table holds only names HAL does not already define, each once.
    #[test]
    fn legacy_aliases_do_not_shadow_hal_names() {
        for (i, &(name, pf)) in LEGACY_FORMAT_ALIASES.iter().enumerate() {
            assert_eq!(PixelFormat::from_str_code(name), None, "{name}");
            assert_eq!(pixel_format_from_wire(name).unwrap(), pf, "{name}");
            assert!(
                LEGACY_FORMAT_ALIASES[..i].iter().all(|&(n, _)| n != name),
                "duplicate alias {name}"
            );
        }
    }

    #[test]
    fn rejects_unknown_format_names() {
        for name in [
            "",
            "RGB",
            "Y800",
            "yuyv",
            "nv12",
            "RGB8",
            "Rgb",
            "ABGR",
            "rgb8_planar_nchw",
        ] {
            let err = pixel_format_from_wire(name).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidData, "{name}");
        }
    }

    #[test]
    fn maps_pixel_formats_to_g2d() {
        assert_eq!(
            pixel_format_to_g2d_format(PixelFormat::Yuyv).unwrap(),
            g2d_format_G2D_YUYV
        );
        assert_eq!(
            pixel_format_to_g2d_format(PixelFormat::Rgba).unwrap(),
            g2d_format_G2D_RGBA8888
        );
        assert_eq!(
            pixel_format_to_g2d_format(PixelFormat::Rgb).unwrap(),
            g2d_format_G2D_RGB888
        );
        assert_eq!(
            pixel_format_to_g2d_format(PixelFormat::Nv12).unwrap(),
            g2d_format_G2D_NV12
        );
        for pf in [PixelFormat::Vyuy, PixelFormat::Bgra, PixelFormat::Nv16] {
            let err = pixel_format_to_g2d_format(pf).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::Unsupported, "{pf:?}");
        }
    }

    #[test]
    fn image_size_matches_format_geometry() {
        assert_eq!(image_size(640, 480, PixelFormat::Rgb), 640 * 480 * 3);
        assert_eq!(image_size(640, 480, PixelFormat::Rgba), 640 * 480 * 4);
        assert_eq!(image_size(640, 480, PixelFormat::Yuyv), 640 * 480 * 2);
        assert_eq!(image_size(640, 480, PixelFormat::Nv12), 640 * 480 * 3 / 2);
    }
}
