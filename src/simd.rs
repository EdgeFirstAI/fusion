// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

//! NEON-accelerated math functions for aarch64.
//!
//! Provides 4-wide f32 SIMD implementations of sincos, atan2, and batch
//! transform operations. On non-aarch64 targets, scalar fallbacks are used.
//!
//! The sincos implementation is based on the Cephes minimax polynomial
//! approach from [Geolm/math_intrinsics](https://github.com/Geolm/math_intrinsics),
//! licensed under MIT. Accuracy: max error ~5.96e-8 (~1 ULP for f32).

/// Compute `(sin(x), cos(x))` for each element of a 4-wide f32 slice.
///
/// `src` contains the input angles in radians.
/// `dst_sin` and `dst_cos` receive the results.
/// All three slices must have the same length. The function processes
/// 4 elements at a time; any remainder is handled with scalar math.
#[allow(dead_code)]
pub fn sincos_f32(src: &[f32], dst_sin: &mut [f32], dst_cos: &mut [f32]) {
    let n = src.len();
    debug_assert_eq!(n, dst_sin.len());
    debug_assert_eq!(n, dst_cos.len());

    #[cfg(target_arch = "aarch64")]
    {
        let chunks = n / 4;
        for c in 0..chunks {
            let i = c * 4;
            let (s, co) = unsafe { neon_sincos_f32x4(src, i) };
            dst_sin[i..i + 4].copy_from_slice(&s);
            dst_cos[i..i + 4].copy_from_slice(&co);
        }
        for i in (chunks * 4)..n {
            let (s, c) = src[i].sin_cos();
            dst_sin[i] = s;
            dst_cos[i] = c;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            let (s, c) = src[i].sin_cos();
            dst_sin[i] = s;
            dst_cos[i] = c;
        }
    }
}

/// Compute `atan2(y, x)` for each element of 4-wide f32 slices.
///
/// Returns results in radians in `dst`. All slices must have the same length.
/// Matches `f32::atan2` at the origin and for signed zeros: `atan2(0.0, 0.0)`
/// is `0.0`, not NaN.
pub fn atan2_f32(y: &[f32], x: &[f32], dst: &mut [f32]) {
    let n = y.len();
    debug_assert_eq!(n, x.len());
    debug_assert_eq!(n, dst.len());

    #[cfg(target_arch = "aarch64")]
    {
        let chunks = n / 4;
        for c in 0..chunks {
            let i = c * 4;
            let result = unsafe { neon_atan2_f32x4(y, x, i) };
            dst[i..i + 4].copy_from_slice(&result);
        }
        for i in (chunks * 4)..n {
            dst[i] = y[i].atan2(x[i]);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            dst[i] = y[i].atan2(x[i]);
        }
    }
}

/// Compute `sqrt(x*x + y*y + z*z)` for 4-wide f32 slices.
///
/// Results stored in `dst`. All slices must have the same length.
pub fn magnitude3_f32(x: &[f32], y: &[f32], z: &[f32], dst: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(n, y.len());
    debug_assert_eq!(n, z.len());
    debug_assert_eq!(n, dst.len());

    #[cfg(target_arch = "aarch64")]
    {
        let chunks = n / 4;
        for c in 0..chunks {
            let i = c * 4;
            let result = unsafe { neon_magnitude3_f32x4(x, y, z, i) };
            dst[i..i + 4].copy_from_slice(&result);
        }
        for i in (chunks * 4)..n {
            dst[i] = (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt();
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            dst[i] = (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt();
        }
    }
}

// ---------------------------------------------------------------------------
// NEON intrinsic implementations (aarch64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// 4-wide sincos using Cephes minimax polynomials.
///
/// Range reduction: 3-constant Cody-Waite (DP1+DP2+DP3 = pi/4).
/// Sin: degree-5 polynomial in x^2, multiplied by x.
/// Cos: degree-6 polynomial in x^2.
/// Branchless quadrant selection via vbslq.
///
/// Coefficients are the Cephes values rounded to f32 and written at the
/// shortest precision that round-trips to the same bits.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_sincos_f32x4(src: &[f32], offset: usize) -> ([f32; 4], [f32; 4]) {
    let x = vld1q_f32(src.as_ptr().add(offset));

    // Constants
    let sign_mask = vdupq_n_u32(0x8000_0000);
    let inv_sign_mask = vdupq_n_u32(0x7FFF_FFFF);
    let four_over_pi = vdupq_n_f32(1.2732395_f32);
    let dp1 = vdupq_n_f32(-0.78515625_f32);
    let dp2 = vdupq_n_f32(-2.4187565e-4_f32);
    let dp3 = vdupq_n_f32(-3.774895e-8_f32);

    // Strip sign, save for later
    let sign_bit_sin = vandq_u32(vreinterpretq_u32_f32(x), sign_mask);
    let xa = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), inv_sign_mask));

    // Range reduction: y = round_even(|x| * 4/pi)
    let mut y = vmulq_f32(xa, four_over_pi);
    let mut emm2 = vcvtq_s32_f32(y);
    emm2 = vaddq_s32(emm2, vdupq_n_s32(1));
    emm2 = vandq_s32(emm2, vdupq_n_s32(!1i32));
    y = vcvtq_f32_s32(emm2);

    // Extended precision modular arithmetic
    let mut xr = vfmaq_f32(xa, y, dp1);
    xr = vfmaq_f32(xr, y, dp2);
    xr = vfmaq_f32(xr, y, dp3);

    // Quadrant selection
    let emm0 = vshlq_n_s32::<29>(vandq_s32(emm2, vdupq_n_s32(4)));
    let swap_sign_bit_sin = vreinterpretq_u32_s32(emm0);

    let poly_mask = vceqq_s32(vandq_s32(emm2, vdupq_n_s32(2)), vdupq_n_s32(0));

    let sign_bit_sin_final = veorq_u32(sign_bit_sin, swap_sign_bit_sin);

    // Cos sign: bit 2 of !(emm2 - 2), shifted into the sign position.
    // With emm2 = 2*k for quadrant index k, the selected polynomial must be
    // negated exactly when k mod 4 is 1 or 2, and bit 2 of !(emm2 - 2) is
    // that predicate. Taking bit 1 instead (`& 2` with a <<30 shift) agrees
    // only for k mod 4 in {0, 1}, inverting cos over |x| mod 2*pi in
    // [3*pi/4, 7*pi/4) while leaving sin correct -- see
    // test_sincos_sweep_all_quadrants.
    let emm2_minus2 = vsubq_s32(emm2, vdupq_n_s32(2));
    let sign_bit_cos = vreinterpretq_u32_s32(vshlq_n_s32::<29>(vandq_s32(
        vmvnq_s32(emm2_minus2),
        vdupq_n_s32(4),
    )));

    let z = vmulq_f32(xr, xr);

    // Cosine polynomial: 1 - z/2 + z^2*(c0 + c1*z + c2*z^2)
    let mut ycos = vdupq_n_f32(2.4433157e-5_f32);
    ycos = vfmaq_f32(vdupq_n_f32(-1.3887316e-3_f32), ycos, z);
    ycos = vfmaq_f32(vdupq_n_f32(4.1666646e-2_f32), ycos, z);
    ycos = vmulq_f32(ycos, z);
    ycos = vmulq_f32(ycos, z);
    ycos = vsubq_f32(ycos, vmulq_f32(z, vdupq_n_f32(0.5_f32)));
    ycos = vaddq_f32(ycos, vdupq_n_f32(1.0_f32));

    // Sine polynomial: x + x*z*(s0 + s1*z + s2*z^2)
    let mut ysin = vdupq_n_f32(-1.9515296e-4_f32);
    ysin = vfmaq_f32(vdupq_n_f32(8.332161e-3_f32), ysin, z);
    ysin = vfmaq_f32(vdupq_n_f32(-1.6666655e-1_f32), ysin, z);
    ysin = vmulq_f32(ysin, z);
    ysin = vfmaq_f32(xr, ysin, xr);

    // Select correct polynomial per lane based on quadrant
    // sin result: where poly_mask is set use ysin, else ycos
    let s1 = vbslq_f32(poly_mask, ysin, ycos);
    // cos result: where poly_mask is set use ycos, else ysin
    let c1 = vbslq_f32(poly_mask, ycos, ysin);

    // Apply signs
    let s_out = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(s1), sign_bit_sin_final));
    let c_out = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(c1), sign_bit_cos));

    let mut s_arr = [0.0f32; 4];
    let mut c_arr = [0.0f32; 4];
    vst1q_f32(s_arr.as_mut_ptr(), s_out);
    vst1q_f32(c_arr.as_mut_ptr(), c_out);
    (s_arr, c_arr)
}

/// 4-wide atan2 using 3-region reduction + degree-4 minimax polynomial.
///
/// Based on Cephes atan implementation from Geolm/math_intrinsics.
/// Max error ~6.7e-5 (sufficient for angle binning).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_atan2_f32x4(y_src: &[f32], x_src: &[f32], offset: usize) -> [f32; 4] {
    let y = vld1q_f32(y_src.as_ptr().add(offset));
    let x = vld1q_f32(x_src.as_ptr().add(offset));

    let sign_mask = vdupq_n_u32(0x8000_0000);
    let abs_mask = vdupq_n_u32(0x7FFF_FFFF);

    let abs_x = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), abs_mask));
    let abs_y = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(y), abs_mask));

    // Swap so we always compute atan of the smaller/larger ratio
    let swap = vcgtq_f32(abs_y, abs_x);
    let max_val = vbslq_f32(swap, abs_y, abs_x);
    let min_val = vbslq_f32(swap, abs_x, abs_y);

    // a = min / max (always in [0, 1]). At the origin both are zero and the
    // division would yield NaN, poisoning the whole lane, so substitute a 1.0
    // divisor there to get a = 0; the sign fixups below then reproduce std's
    // atan2(+-0.0, +-0.0) exactly.
    let denom = vbslq_f32(
        vceqq_f32(max_val, vdupq_n_f32(0.0_f32)),
        vdupq_n_f32(1.0_f32),
        max_val,
    );
    let a = vdivq_f32(min_val, denom);
    let a2 = vmulq_f32(a, a);

    // Degree-4 minimax polynomial for atan(a) on [0, 1]
    // atan(a) ~= a * (c0 + c1*a^2 + c2*a^4 + c3*a^6 + c4*a^8)
    let mut p = vdupq_n_f32(0.0028662257_f32);
    p = vfmaq_f32(vdupq_n_f32(-0.016165737_f32), p, a2);
    p = vfmaq_f32(vdupq_n_f32(0.042909615_f32), p, a2);
    p = vfmaq_f32(vdupq_n_f32(-0.07528964_f32), p, a2);
    p = vfmaq_f32(vdupq_n_f32(0.10656264_f32), p, a2);
    p = vfmaq_f32(vdupq_n_f32(-0.142089_f32), p, a2);
    p = vfmaq_f32(vdupq_n_f32(0.19993551_f32), p, a2);
    p = vfmaq_f32(vdupq_n_f32(-0.33333147_f32), p, a2);
    let atan_a = vfmaq_f32(a, vmulq_f32(p, a2), a);

    // If swapped, result = pi/2 - atan_a
    let pi_over_2 = vdupq_n_f32(std::f32::consts::FRAC_PI_2);
    let result = vbslq_f32(swap, vsubq_f32(pi_over_2, atan_a), atan_a);

    // Apply sign of x: if x is negative, result = pi - result. Test x's sign
    // bit rather than `x < 0.0` so that x = -0.0 counts as negative, which is
    // what atan2(+-0.0, -0.0) = +-pi requires. For every other input the two
    // tests agree.
    let x_neg = vcltq_s32(vreinterpretq_s32_f32(x), vdupq_n_s32(0));
    let pi = vdupq_n_f32(std::f32::consts::PI);
    let result = vbslq_f32(x_neg, vsubq_f32(pi, result), result);

    // Apply sign of y
    let y_sign = vandq_u32(vreinterpretq_u32_f32(y), sign_mask);
    let result = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(result), y_sign));

    let mut out = [0.0f32; 4];
    vst1q_f32(out.as_mut_ptr(), result);
    out
}

/// 4-wide 3D magnitude: sqrt(x^2 + y^2 + z^2)
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_magnitude3_f32x4(xs: &[f32], ys: &[f32], zs: &[f32], offset: usize) -> [f32; 4] {
    let x = vld1q_f32(xs.as_ptr().add(offset));
    let y = vld1q_f32(ys.as_ptr().add(offset));
    let z = vld1q_f32(zs.as_ptr().add(offset));

    let sum = vfmaq_f32(vfmaq_f32(vmulq_f32(x, x), y, y), z, z);
    let result = vsqrtq_f32(sum);

    let mut out = [0.0f32; 4];
    vst1q_f32(out.as_mut_ptr(), result);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sincos_basic() {
        let angles = [
            0.0_f32,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            1.0,
        ];
        let mut s = [0.0f32; 4];
        let mut c = [0.0f32; 4];
        sincos_f32(&angles, &mut s, &mut c);

        for i in 0..4 {
            let (expected_s, expected_c) = angles[i].sin_cos();
            assert!(
                (s[i] - expected_s).abs() < 1e-6,
                "sin({}) = {} expected {}",
                angles[i],
                s[i],
                expected_s
            );
            assert!(
                (c[i] - expected_c).abs() < 1e-6,
                "cos({}) = {} expected {}",
                angles[i],
                c[i],
                expected_c
            );
        }
    }

    #[test]
    fn test_sincos_negative_and_large() {
        let angles = [-1.0_f32, -3.0, 10.0, -100.0, 0.5, 2.5, -0.1, 7.0];
        let mut s = vec![0.0f32; 8];
        let mut c = vec![0.0f32; 8];
        sincos_f32(&angles, &mut s, &mut c);

        for i in 0..8 {
            let (expected_s, expected_c) = angles[i].sin_cos();
            assert!(
                (s[i] - expected_s).abs() < 1e-5,
                "sin({}) = {} expected {}",
                angles[i],
                s[i],
                expected_s
            );
            assert!(
                (c[i] - expected_c).abs() < 1e-5,
                "cos({}) = {} expected {}",
                angles[i],
                c[i],
                expected_c
            );
        }
    }

    #[test]
    fn test_sincos_remainder() {
        // 5 elements: 4 via SIMD + 1 scalar remainder
        let angles = [0.1_f32, 0.2, 0.3, 0.4, 0.5];
        let mut s = vec![0.0f32; 5];
        let mut c = vec![0.0f32; 5];
        sincos_f32(&angles, &mut s, &mut c);

        for i in 0..5 {
            let (expected_s, expected_c) = angles[i].sin_cos();
            assert!(
                (s[i] - expected_s).abs() < 1e-6,
                "sin({}) = {} expected {}",
                angles[i],
                s[i],
                expected_s
            );
            assert!(
                (c[i] - expected_c).abs() < 1e-6,
                "cos({}) = {} expected {}",
                angles[i],
                c[i],
                expected_c
            );
        }
    }

    /// Dense sweep across every quadrant of the Cephes range reduction.
    ///
    /// The narrow smoke tests above cluster near the first octant, where a
    /// wrong quadrant sign mask still produces correct results. Cover several
    /// full turns in both directions so any quadrant-selection or sign error
    /// is caught, and report the first offending angle rather than the last.
    #[test]
    fn test_sincos_sweep_all_quadrants() {
        const N: usize = 4096;
        const SPAN: f32 = 8.0 * std::f32::consts::PI;

        let angles: Vec<f32> = (0..N)
            .map(|i| -0.5 * SPAN + (i as f32) * SPAN / (N as f32))
            .collect();
        let mut s = vec![0.0f32; N];
        let mut c = vec![0.0f32; N];
        sincos_f32(&angles, &mut s, &mut c);

        for i in 0..N {
            let (expected_s, expected_c) = angles[i].sin_cos();
            assert!(
                (s[i] - expected_s).abs() < 1e-5,
                "sin({}) = {} expected {}",
                angles[i],
                s[i],
                expected_s
            );
            assert!(
                (c[i] - expected_c).abs() < 1e-5,
                "cos({}) = {} expected {}",
                angles[i],
                c[i],
                expected_c
            );
        }
    }

    #[test]
    fn test_atan2_basic() {
        let y = [0.0_f32, 1.0, -1.0, 1.0];
        let x = [1.0_f32, 0.0, 0.0, 1.0];
        let mut dst = [0.0f32; 4];
        atan2_f32(&y, &x, &mut dst);

        for i in 0..4 {
            let expected = y[i].atan2(x[i]);
            assert!(
                (dst[i] - expected).abs() < 1e-4,
                "atan2({}, {}) = {} expected {}",
                y[i],
                x[i],
                dst[i],
                expected
            );
        }
    }

    #[test]
    fn test_atan2_all_quadrants() {
        let y = [1.0_f32, 1.0, -1.0, -1.0, 0.5, -0.5, 3.0, -3.0];
        let x = [1.0_f32, -1.0, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1];
        let mut dst = vec![0.0f32; 8];
        atan2_f32(&y, &x, &mut dst);

        for i in 0..8 {
            let expected = y[i].atan2(x[i]);
            assert!(
                (dst[i] - expected).abs() < 1e-3,
                "atan2({}, {}) = {} expected {}",
                y[i],
                x[i],
                dst[i],
                expected
            );
        }
    }

    /// atan2 at the origin and across signed zeros.
    ///
    /// The NEON path divides min/max, which is 0/0 at the origin and returned
    /// NaN before the divisor was guarded. Signed zeros additionally pin the
    /// "x is negative" test to x's sign bit, since -0.0 is not less than 0.0
    /// yet atan2(+-0.0, -0.0) is +-pi.
    #[test]
    fn test_atan2_origin_and_signed_zeros() {
        let y = [0.0_f32, -0.0, 0.0, -0.0, 0.0, -0.0, 1.0, -1.0];
        let x = [0.0_f32, 0.0, -0.0, -0.0, 1.0, -1.0, 0.0, -0.0];
        let mut dst = vec![0.0f32; 8];
        atan2_f32(&y, &x, &mut dst);

        for i in 0..8 {
            let expected = y[i].atan2(x[i]);
            assert!(
                dst[i].is_finite(),
                "atan2({}, {}) = {} is not finite",
                y[i],
                x[i],
                dst[i]
            );
            assert!(
                (dst[i] - expected).abs() < 1e-4,
                "atan2({}, {}) = {} expected {}",
                y[i],
                x[i],
                dst[i],
                expected
            );
        }
    }

    #[test]
    fn test_magnitude3_basic() {
        let x = [3.0_f32, 0.0, 1.0, 2.0];
        let y = [4.0_f32, 0.0, 1.0, 3.0];
        let z = [0.0_f32, 0.0, 1.0, 6.0];
        let mut dst = [0.0f32; 4];
        magnitude3_f32(&x, &y, &z, &mut dst);

        let expected = [5.0, 0.0, 3.0_f32.sqrt(), 7.0];
        for i in 0..4 {
            assert!(
                (dst[i] - expected[i]).abs() < 1e-6,
                "mag({},{},{}) = {} expected {}",
                x[i],
                y[i],
                z[i],
                dst[i],
                expected[i]
            );
        }
    }
}
