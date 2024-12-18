#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

include!("ffi.rs");

use std::os::fd::AsRawFd;

use dma_buf::DmaBuf;
use nix::ioctl_write_ptr;

use videostream::fourcc::FourCC;

const RGB3: FourCC = FourCC(*b"RGB3");
const RGBX: FourCC = FourCC(*b"RGBX");
const RGBA: FourCC = FourCC(*b"RGBA");
const YUYV: FourCC = FourCC(*b"YUYV");
const NV12: FourCC = FourCC(*b"NV12");

pub struct G2DFormat(g2d_format);

impl G2DFormat {
    pub fn from(fourcc: FourCC) -> Self {
        fourcc.into()
    }

    pub fn format(&self) -> g2d_format {
        self.0
    }
}

impl From<FourCC> for G2DFormat {
    fn from(format: FourCC) -> Self {
        match format {
            RGB3 => G2DFormat(g2d_format_G2D_RGB888),
            RGBX => G2DFormat(g2d_format_G2D_RGBX8888),
            RGBA => G2DFormat(g2d_format_G2D_RGBA8888),
            YUYV => G2DFormat(g2d_format_G2D_YUYV),
            NV12 => G2DFormat(g2d_format_G2D_NV12),
            _ => todo!(),
        }
    }
}

impl From<G2DFormat> for FourCC {
    fn from(format: G2DFormat) -> Self {
        match format.0 {
            g2d_format_G2D_RGB888 => RGB3,
            g2d_format_G2D_RGBX8888 => RGBX,
            g2d_format_G2D_RGBA8888 => RGBA,
            g2d_format_G2D_YUYV => YUYV,
            g2d_format_G2D_NV12 => NV12,
            _ => todo!(),
        }
    }
}

pub struct G2DPhysical(::std::os::raw::c_int);

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct dma_buf_phys(std::ffi::c_ulong);

const DMA_BUF_BASE: u8 = b'b';
const DMA_BUF_IOCTL_PHYS: u8 = 10;
ioctl_write_ptr!(
    ioctl_dma_buf_phys,
    DMA_BUF_BASE,
    DMA_BUF_IOCTL_PHYS,
    std::ffi::c_ulong
);

impl From<DmaBuf> for G2DPhysical {
    fn from(buf: DmaBuf) -> Self {
        let phys = dma_buf_phys(0);
        let err = unsafe { ioctl_dma_buf_phys(buf.as_raw_fd(), &phys.0).unwrap_or(0) };
        if err != 0 {
            return G2DPhysical(0);
        }

        G2DPhysical(phys.0 as i32)
    }
}

impl From<i32> for G2DPhysical {
    fn from(buf: i32) -> Self {
        G2DPhysical(buf)
    }
}

impl From<G2DPhysical> for i32 {
    fn from(phys: G2DPhysical) -> Self {
        phys.0
    }
}
