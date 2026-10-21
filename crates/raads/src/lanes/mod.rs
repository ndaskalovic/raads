//! SIMD packs: W lanes of `f32` or `f64`, one independent evaluation per lane.
//!
//! One implementation on `std::simd` for every target ([`F32x4`] … [`F64x16`]), plus plain `f64`
//! for one lane. LLVM chooses the instructions from the enabled target features (see the crate
//! docs for build flags). Engines and generated kernels are written against [`Real`].

use crate::Real;
use std::fmt::Debug;

/// `$a·$b + $c` for the scalar backend: fused where the target has an FMA instruction
/// (aarch64, x86_64 with `fma`), else two rounded ops, because std `mul_add` is a libm call there.
macro_rules! mul_add {
    ($a:expr, $b:expr, $c:expr) => {{
        #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
        let r = $a.mul_add($b, $c);
        #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
        let r = $a * $b + $c;
        r
    }};
}

mod scalar;
mod simd;

impl<const W: usize> crate::sealed::Sealed for simd::F32<W> {}
impl<const W: usize> crate::sealed::Sealed for simd::F64<W> {}

/// A pack of `W` floats. Also a first-order [`Real`] (its own value, zero tangent).
/// Sealed through [`AdNum`](crate::AdNum).
pub trait Lanes: Real<Lanes = Self> + Debug + PartialEq {
    /// `f32` or `f64`.
    type Elem: Copy + Debug + Default + PartialEq + Into<f64>;
    /// Number of lanes.
    const W: usize;
    /// First `W` elements of `src`.
    fn load(src: &[Self::Elem]) -> Self;
    /// First `W` elements of `src`, converted (e.g. random draws stored as `f32` for every precision).
    fn load_f32(src: &[f32]) -> Self;
    /// First `W` elements of `src`, converted (e.g. random draws stored as `f64` for an `f32` pack).
    fn load_f64(src: &[f64]) -> Self;
    /// Write the `W` lanes to `dst[..W]`.
    fn store(self, dst: &mut [Self::Elem]);
    /// Lane `i` as `f64`.
    fn lane(self, i: usize) -> f64;
    /// Pack with lane `i` = `f(i)`.
    fn from_fn(f: impl FnMut(usize) -> f64) -> Self;
    /// Sum of the lanes, accumulated in `f64`.
    #[inline]
    fn sum(self) -> f64 {
        (0..Self::W).map(|i| self.lane(i)).sum()
    }
}

/// `x^k` for the exponents with a cheaper closed form: `k` = 0, ±1, 2, 3, -2 (multiplies and one
/// reciprocal) and ±0.5 (`sqrt`). `x^±0.5` follows `sqrt` at `-0` and `-inf`: `(-0)^0.5 = -0`,
/// `(-0)^-0.5 = -inf`, `(-inf)^±0.5 = NaN` (`std`: `+0`, `+inf`, `+inf`/`+0`). Used by every
/// pack's `powf`, so the value, the derivative's `x^(k-1)` and `Dual` tangents all take it; `k` is
/// a literal in kernels, so the branch folds away.
#[inline(always)]
pub(crate) fn powf_small<L: Lanes>(x: L, k: f64) -> Option<L> {
    let r = if k == 0.0 {
        L::from(1.0)
    } else if k == 1.0 {
        x
    } else if k == 2.0 {
        x * x
    } else if k == 3.0 {
        x * x * x
    } else if k == -1.0 {
        x.rdiv(1.0)
    } else if k == -2.0 {
        (x * x).rdiv(1.0)
    } else if k == 0.5 {
        x.sqrt()
    } else if k == -0.5 {
        x.sqrt().rdiv(1.0)
    } else {
        return None;
    };
    Some(r)
}

/// Polynomial `exp`: `exp(x) = 2^n · p(r)`, `n = round(x / ln2)`, `r = x - n·ln2` (two-constant
/// Cody-Waite), `p` the Taylor polynomial in FMA Horner form. Max relative error ≈ 2.5e-7
/// (`f32`, degree 6) and ≈ 6e-16 (`f64`, degree 12).
///
/// `x` is clamped to `[LO, HI]` first (NaN passes through). `HI` lies just above `ln(MAX)`, so
/// results are finite up to the true overflow threshold and `+inf` above it. `2^n` is applied by
/// exponent-bit arithmetic, with the coefficients doubled and the scale `2^(n-1)` so that
/// `n = 128` (`f64`: 1024) is representable. The lowest `n` gives an exponent field of 0, i.e. 0:
/// results below `2^-125.5` (`x < -86.99`; `f64`: `2^-1021.5`, `x < -708.05`) flush to 0.
pub(crate) mod exp_poly {
    pub mod f32 {
        pub mod bits {
            pub const LOG2E: f32 = std::f32::consts::LOG2_E;
            pub const LN2_HI: f32 = 0.693_145_75;
            pub const LN2_LO: f32 = 1.428_606_8e-6;
            /// ln(f32::MAX) = 88.72; `n` stays ≤ 128.
            pub const HI: f32 = 89.0;
            /// `n = -126` here (exponent field 0 after the bias of 126).
            pub const LO: f32 = -87.6;
            /// `2·P`: `p(r) = 2e^r`.
            pub const P: [f32; 7] = [2.0 / 720.0, 2.0 / 120.0, 2.0 / 24.0, 2.0 / 6.0, 1.0, 2.0, 2.0];
        }
    }
    pub mod f64 {
        pub mod bits {
            pub const LOG2E: f64 = std::f64::consts::LOG2_E;
            pub const LN2_HI: f64 = 0.693_145_751_953_125;
            pub const LN2_LO: f64 = 1.428_606_820_309_417_3e-6;
            /// ln(f64::MAX) = 709.78; `n` stays ≤ 1024.
            pub const HI: f64 = 710.0;
            /// `n = -1022` here (exponent field 0 after the bias of 1022).
            pub const LO: f64 = -708.7;
            /// `2·P`: `p(r) = 2e^r`.
            pub const P: [f64; 13] = [
                2.0 / 479_001_600.0,
                2.0 / 39_916_800.0,
                2.0 / 3_628_800.0,
                2.0 / 362_880.0,
                2.0 / 40_320.0,
                2.0 / 5_040.0,
                2.0 / 720.0,
                2.0 / 120.0,
                2.0 / 24.0,
                2.0 / 6.0,
                1.0,
                2.0,
                2.0,
            ];
        }
    }
}

/// Polynomial `ln`: `x = 2^e·m` with `m` in `[√2/2, √2)` (exponent-bit arithmetic; subnormals
/// are scaled up first), then `ln(m) = ln(1 + f) = 2·atanh(s)`, `s = f / (2 + f)`, evaluated as
/// `f - (f²/2 - s·(f²/2 + R))` with `R = z·P(z)`, `z = s²`, `P` the Taylor polynomial of the
/// `atanh` series (as in musl's `log`), degree 3 (`f32`) or 8 (`f64`) in `z` (`|s| ≤ 0.172`).
/// Within 1 ulp of `std`. `ln(0) = -inf`, `ln(+inf) = +inf`, `ln(x < 0) = NaN`.
pub(crate) mod ln_poly {
    pub mod f32 {
        pub mod bits {
            /// Bits of √2/2: subtracting them maps `[√2/2, √2)` to exponent 0.
            pub const SQRT_HALF: u32 = 0x3f35_04f3;
            pub const MANT: u32 = (1 << 23) - 1;
            /// Subnormals are multiplied by `SCALE = 2^SCALE_LOG2`.
            pub const SCALE: f32 = 16_777_216.0;
            pub const SCALE_LOG2: f32 = 24.0;
            /// `2/(2k+1)`, highest degree first: `R = 2z/3 + 2z²/5 + …`.
            pub const P: [f32; 4] = [2.0 / 9.0, 2.0 / 7.0, 2.0 / 5.0, 2.0 / 3.0];
        }
    }
    pub mod f64 {
        pub mod bits {
            /// Bits of √2/2: subtracting them maps `[√2/2, √2)` to exponent 0.
            pub const SQRT_HALF: u64 = 0x3fe6_a09e_667f_3bcd;
            pub const MANT: u64 = (1 << 52) - 1;
            /// Subnormals are multiplied by `SCALE = 2^SCALE_LOG2`.
            pub const SCALE: f64 = 18_014_398_509_481_984.0;
            pub const SCALE_LOG2: f64 = 54.0;
            /// `2/(2k+1)`, highest degree first: `R = 2z/3 + 2z²/5 + …`.
            pub const P: [f64; 9] =
                [2.0 / 19.0, 2.0 / 17.0, 2.0 / 15.0, 2.0 / 13.0, 2.0 / 11.0, 2.0 / 9.0, 2.0 / 7.0, 2.0 / 5.0, 2.0 / 3.0];
        }
    }
}

/// 4 lanes of `f32`.
pub type F32x4 = simd::F32<4>;
/// 8 lanes of `f32`.
pub type F32x8 = simd::F32<8>;
/// 16 lanes of `f32`.
pub type F32x16 = simd::F32<16>;
/// 32 lanes of `f32`.
pub type F32x32 = simd::F32<32>;
/// 2 lanes of `f64`.
pub type F64x2 = simd::F64<2>;
/// 4 lanes of `f64`.
pub type F64x4 = simd::F64<4>;
/// 8 lanes of `f64`.
pub type F64x8 = simd::F64<8>;
/// 16 lanes of `f64`.
pub type F64x16 = simd::F64<16>;

/// The registers inside a pack are not public API:
/// ```compile_fail
/// let _ = raads::lanes::F32x4::from(1.0).0;
/// ```
#[cfg(doctest)]
struct PrivatePackFields;

#[cfg(test)]
mod tests;
