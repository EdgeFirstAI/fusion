// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use async_pidfd::PidFd;
use core::fmt;
use dma_heap::{Heap, HeapKind};
use edgefirst_schemas::{edgefirst_msgs::CameraFrame, tensor::TensorPlaneView};
pub use edgefirst_tensor::PixelFormat;
use edgefirst_tensor::PixelLayout;
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

/// Build a [`G2DSurface`] from an [`Image`]'s DMA buffer(s).
fn surface_from_image(img: &Image) -> Result<G2DSurface, Box<dyn Error>> {
    let addr = G2DPhysical::new(img.fd.as_raw_fd())?.address();
    let chroma_addr = match &img.chroma_fd {
        Some(fd) => G2DPhysical::new(fd.as_raw_fd())?.address(),
        None => addr,
    };
    Ok(g2d_surface(
        img.width,
        img.height,
        img.format,
        &img.layout,
        addr,
        chroma_addr,
    )?)
}

/// The [`G2DSurface`] for a `width`x`height` `format` image laid out as
/// `layout`, whose plane 0 buffer starts at physical address `addr` and whose
/// chroma buffer starts at `chroma_addr` (equal to `addr` when both planes
/// share one buffer).
///
/// G2D has no per-plane offset and counts its stride in pixels, so the
/// offsets are folded into the plane addresses and the byte stride must be a
/// whole number of pixels. Both planes of NV12 share that one stride.
fn g2d_surface(
    width: u32,
    height: u32,
    format: PixelFormat,
    layout: &ImageLayout,
    addr: u64,
    chroma_addr: u64,
) -> io::Result<G2DSurface> {
    let g2d_format = pixel_format_to_g2d_format(format)?;
    let invalid = |msg: String| io::Error::new(io::ErrorKind::InvalidInput, msg);
    let bytes_per_pixel = format.channels() as u32;
    if !layout.stride.is_multiple_of(bytes_per_pixel) {
        return Err(invalid(format!(
            "G2D cannot address a {}-byte {} row: not a whole number of {bytes_per_pixel}-byte pixels",
            layout.stride,
            format.as_str()
        )));
    }
    let planes = match (format.layout(), layout.chroma_offset) {
        (PixelLayout::SemiPlanar, Some(chroma)) => [
            addr + u64::from(layout.offset),
            chroma_addr + u64::from(chroma),
            0,
        ],
        (PixelLayout::SemiPlanar, None) => {
            return Err(invalid(format!(
                "{} image has no chroma plane",
                format.as_str()
            )))
        }
        _ => [addr + u64::from(layout.offset), 0, 0],
    };
    let to_i32 = |name: &str, v: u32| {
        i32::try_from(v).map_err(|_| invalid(format!("G2D {name} {v} exceeds i32")))
    };
    let width = to_i32("width", width)?;
    let height = to_i32("height", height)?;
    Ok(G2DSurface {
        planes,
        format: g2d_format,
        left: 0,
        top: 0,
        right: width,
        bottom: height,
        stride: to_i32("stride", layout.stride / bytes_per_pixel)?,
        width,
        height,
        blendfunc: 0,
        clrcolor: 0,
        rot: 0,
        global_alpha: 0,
    })
}

/// Where an [`Image`]'s pixels sit in its DMA-BUF(s), in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageLayout {
    /// Row pitch of plane 0, and of the chroma plane of a semi-planar format.
    pub stride: u32,
    /// Offset of plane 0 in [`Image::fd`].
    pub offset: u32,
    /// Offset of the chroma plane of a semi-planar format: in
    /// [`Image::chroma_fd`] when the image has one, else in [`Image::fd`].
    /// `None` for other formats.
    pub chroma_offset: Option<u32>,
}

impl ImageLayout {
    /// The layout HAL allocates for a `width`x`height` `format` image: rows
    /// of `width` pixels with no padding, planes back to back from offset 0.
    pub fn dense(width: u32, height: u32, format: PixelFormat) -> io::Result<Self> {
        let stride = min_row_bytes(width, format)?;
        Self::contiguous(width, height, format, stride, 0)
    }

    /// The layout of a single buffer whose plane 0 starts at `offset` with
    /// rows of `stride` bytes, the chroma plane (if any) following it as HAL's
    /// [`PixelFormat::plane_table`] places it.
    fn contiguous(
        width: u32,
        height: u32,
        format: PixelFormat,
        stride: u32,
        offset: u32,
    ) -> io::Result<Self> {
        let chroma_offset = match format.layout() {
            PixelLayout::SemiPlanar => {
                let chroma = format
                    .plane_table(width as usize, height as usize, stride as usize)
                    .and_then(|planes| planes.get(1).map(|p| p.offset))
                    .and_then(|o| o.checked_add(u64::from(offset)))
                    .ok_or_else(|| {
                        camera_frame_invalid(format!(
                            "no chroma plane for a {width}x{height} {} image at stride {stride}",
                            format.as_str()
                        ))
                    })?;
                Some(camera_frame_u32("chroma offset", chroma)?)
            }
            _ => None,
        };
        Ok(Self {
            stride,
            offset,
            chroma_offset,
        })
    }
}

/// Bytes in one unpadded row of plane 0 of a `width`-pixel `format` image.
fn min_row_bytes(width: u32, format: PixelFormat) -> io::Result<u32> {
    width.checked_mul(format.channels() as u32).ok_or_else(|| {
        camera_frame_invalid(format!(
            "a {width}-pixel {} row exceeds u32 bytes",
            format.as_str()
        ))
    })
}

pub struct Image {
    pub fd: OwnedFd,
    /// The buffer holding the chroma plane, when it is not [`Image::fd`].
    pub chroma_fd: Option<OwnedFd>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub layout: ImageLayout,
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("fd", &self.fd)
            .field("chroma_fd", &self.chroma_fd)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format.as_str())
            .field("layout", &self.layout)
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
        let layout = ImageLayout::dense(width, height, format)?;
        let heap = Heap::new(HeapKind::Cma)?;
        let fd = heap.allocate(size)?;
        Ok(Self {
            fd,
            chroma_fd: None,
            width,
            height,
            format,
            layout,
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

/// The layout of an imported frame, from its plane table.
///
/// Plane 0's `stride` is the row pitch; 0 means unpadded rows. For a
/// semi-planar format, plane 1 is the chroma plane, at its own `offset` in its
/// own buffer when its `handle` differs from plane 0's. Without a plane 1 the
/// chroma plane follows plane 0 in the same buffer, as camera 2.x publishes
/// NV12. Returns the layout and, for a chroma plane in another buffer, that
/// buffer's handle.
fn camera_frame_layout(
    width: u32,
    height: u32,
    format: PixelFormat,
    plane0: &TensorPlaneView<'_>,
    plane1: Option<&TensorPlaneView<'_>>,
) -> io::Result<(ImageLayout, Option<i64>)> {
    let min_stride = min_row_bytes(width, format)?;
    let stride = match camera_frame_u32("stride", plane0.stride)? {
        0 => min_stride,
        s if s < min_stride => {
            return Err(camera_frame_invalid(format!(
                "CameraFrame plane 0 stride {s} is shorter than a {width}-pixel {} row ({min_stride} bytes)",
                format.as_str()
            )))
        }
        s => s,
    };
    let offset = camera_frame_u32("offset", plane0.offset)?;
    let mut layout = ImageLayout::contiguous(width, height, format, stride, offset)?;
    let chroma = match (format.layout(), plane1) {
        (PixelLayout::SemiPlanar, Some(p)) => p,
        _ => return Ok((layout, None)),
    };
    if chroma.stride != 0 && chroma.stride != u64::from(stride) {
        return Err(camera_frame_invalid(format!(
            "CameraFrame {} chroma stride {} differs from luma stride {stride}",
            format.as_str(),
            chroma.stride
        )));
    }
    layout.chroma_offset = Some(camera_frame_u32("chroma offset", chroma.offset)?);
    let chroma_handle = (chroma.handle != plane0.handle).then_some(chroma.handle);
    Ok((layout, chroma_handle))
}

/// Duplicate the producer's DMA-BUF `handle` into this process.
fn import_plane_fd(pidfd: &PidFd, handle: i64) -> io::Result<OwnedFd> {
    let target_fd = i32::try_from(handle).map_err(|_| {
        camera_frame_invalid(format!("CameraFrame plane handle {handle} exceeds i32"))
    })?;
    Ok(get_file_from_pidfd(pidfd.as_raw_fd(), target_fd, GetFdFlags::empty())?.into())
}

/// Import a [`CameraFrame`] tensor as an [`Image`] via pidfd + getfd, keeping
/// the producer's row stride and plane offsets.
pub fn image_from_camera_frame(frame: &CameraFrame<Vec<u8>>) -> Result<Image, io::Error> {
    let t = frame.tensor();
    let plane0 = t
        .plane_at(0)
        .ok_or_else(|| camera_frame_invalid("CameraFrame tensor has no plane 0"))?;
    let height = t
        .shape_at(0)
        .ok_or_else(|| camera_frame_invalid("CameraFrame tensor missing height (shape[0])"))?;
    let width = t
        .shape_at(1)
        .ok_or_else(|| camera_frame_invalid("CameraFrame tensor missing width (shape[1])"))?;
    let width = camera_frame_nonzero_u32("width", width)?;
    let height = camera_frame_nonzero_u32("height", height)?;
    let format = pixel_format_from_wire(t.format())?;
    let (layout, chroma_handle) =
        camera_frame_layout(width, height, format, &plane0, t.plane_at(1).as_ref())?;
    let pid = t.pid();
    let pid_i32 = i32::try_from(pid)
        .map_err(|_| camera_frame_invalid(format!("CameraFrame tensor pid {pid} exceeds i32")))?;
    let pidfd: PidFd = PidFd::from_pid(pid_i32)?;
    let fd = import_plane_fd(&pidfd, plane0.handle)?;
    let chroma_fd = chroma_handle
        .map(|handle| import_plane_fd(&pidfd, handle))
        .transpose()?;
    Ok(Image {
        fd,
        chroma_fd,
        width,
        height,
        format,
        layout,
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

    fn plane(handle: i64, offset: u64, stride: u64) -> TensorPlaneView<'static> {
        TensorPlaneView {
            handle,
            offset,
            stride,
            size: 0,
            used: 0,
            modifier: 0,
            handle_bytes: &[],
            data: &[],
        }
    }

    const BASE: u64 = 0x4000_0000;
    const CHROMA_BASE: u64 = 0x5000_0000;

    /// A padded YUYV row, as the camera SDK publishes it: 640 pixels in a
    /// 1536-byte row become a 768-pixel G2D stride.
    #[test]
    fn padded_yuyv_keeps_the_producer_stride() {
        let (layout, chroma) =
            camera_frame_layout(640, 480, PixelFormat::Yuyv, &plane(3, 0, 1536), None).unwrap();
        assert_eq!(
            layout,
            ImageLayout {
                stride: 1536,
                offset: 0,
                chroma_offset: None
            }
        );
        assert_eq!(chroma, None);
        let s = g2d_surface(640, 480, PixelFormat::Yuyv, &layout, BASE, BASE).unwrap();
        assert_eq!(s.planes, [BASE, 0, 0]);
        assert_eq!(s.stride, 768);
        assert_eq!((s.width, s.height, s.right, s.bottom), (640, 480, 640, 480));
        assert_eq!(s.format, g2d_format_G2D_YUYV);
    }

    /// Single-buffer NV12 with a padded pitch and an aligned chroma plane:
    /// the plane offsets are used as published, not derived.
    #[test]
    fn padded_nv12_uses_the_published_plane_offsets() {
        let (stride, y_offset, uv_offset) = (2048, 4096, 4096 + 2048 * 1088);
        let (layout, chroma) = camera_frame_layout(
            1920,
            1080,
            PixelFormat::Nv12,
            &plane(7, y_offset, stride),
            Some(&plane(7, uv_offset, stride)),
        )
        .unwrap();
        assert_eq!(chroma, None);
        let s = g2d_surface(1920, 1080, PixelFormat::Nv12, &layout, BASE, BASE).unwrap();
        assert_eq!(s.planes, [BASE + y_offset, BASE + uv_offset, 0]);
        assert_eq!(s.stride, 2048);
        assert_eq!((s.width, s.height), (1920, 1080));
        assert_eq!(s.format, g2d_format_G2D_NV12);
    }

    /// NV12M: the chroma plane lives in its own DMA-BUF, at its own offset.
    #[test]
    fn nv12m_chroma_comes_from_its_own_buffer() {
        let (layout, chroma) = camera_frame_layout(
            1920,
            1080,
            PixelFormat::Nv12,
            &plane(70, 0, 1920),
            Some(&plane(71, 0, 1920)),
        )
        .unwrap();
        assert_eq!(chroma, Some(71));
        let s = g2d_surface(1920, 1080, PixelFormat::Nv12, &layout, BASE, CHROMA_BASE).unwrap();
        assert_eq!(s.planes, [BASE, CHROMA_BASE, 0]);
        assert_eq!(s.stride, 1920);
    }

    /// Camera 2.x publishes NV12 as plane 0 only; chroma follows the luma
    /// rows at the published pitch.
    #[test]
    fn nv12_without_a_chroma_plane_follows_plane_0() {
        let (layout, chroma) =
            camera_frame_layout(640, 480, PixelFormat::Nv12, &plane(7, 256, 704), None).unwrap();
        assert_eq!(chroma, None);
        assert_eq!(layout.chroma_offset, Some(256 + 704 * 480));
        let s = g2d_surface(640, 480, PixelFormat::Nv12, &layout, BASE, BASE).unwrap();
        assert_eq!(s.planes, [BASE + 256, BASE + 256 + 704 * 480, 0]);
        assert_eq!(s.stride, 704);
    }

    /// A zero stride means unpadded rows.
    #[test]
    fn zero_stride_is_dense() {
        let (layout, _) =
            camera_frame_layout(640, 480, PixelFormat::Rgb, &plane(3, 0, 0), None).unwrap();
        assert_eq!(
            layout,
            ImageLayout::dense(640, 480, PixelFormat::Rgb).unwrap()
        );
        assert_eq!(layout.stride, 640 * 3);
    }

    /// Owned images are dense: unpadded rows, chroma straight after luma.
    #[test]
    fn owned_layouts_are_dense() {
        for (format, stride, chroma) in [
            (PixelFormat::Rgba, 640 * 4, None),
            (PixelFormat::Yuyv, 640 * 2, None),
            (PixelFormat::Nv12, 640, Some(640 * 480)),
        ] {
            let layout = ImageLayout::dense(640, 480, format).unwrap();
            assert_eq!(
                layout,
                ImageLayout {
                    stride,
                    offset: 0,
                    chroma_offset: chroma
                },
                "{format:?}"
            );
            let s = g2d_surface(640, 480, format, &layout, BASE, BASE).unwrap();
            assert_eq!(s.stride, 640, "{format:?}");
        }
    }

    #[test]
    fn rejects_layouts_g2d_cannot_express() {
        let short = camera_frame_layout(640, 480, PixelFormat::Yuyv, &plane(3, 0, 1279), None);
        assert_eq!(short.unwrap_err().kind(), io::ErrorKind::InvalidData);

        let split_stride = camera_frame_layout(
            640,
            480,
            PixelFormat::Nv12,
            &plane(7, 0, 640),
            Some(&plane(7, 640 * 480, 704)),
        );
        assert_eq!(split_stride.unwrap_err().kind(), io::ErrorKind::InvalidData);

        let (odd, _) =
            camera_frame_layout(640, 480, PixelFormat::Yuyv, &plane(3, 0, 1537), None).unwrap();
        let err = g2d_surface(640, 480, PixelFormat::Yuyv, &odd, BASE, BASE).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn image_size_matches_format_geometry() {
        assert_eq!(image_size(640, 480, PixelFormat::Rgb), 640 * 480 * 3);
        assert_eq!(image_size(640, 480, PixelFormat::Rgba), 640 * 480 * 4);
        assert_eq!(image_size(640, 480, PixelFormat::Yuyv), 640 * 480 * 2);
        assert_eq!(image_size(640, 480, PixelFormat::Nv12), 640 * 480 * 3 / 2);
    }
}
