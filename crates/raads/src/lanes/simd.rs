//! Packs on `std::simd`: `F32<W>` = `Simd<f32, W>`, `F64<W>` = `Simd<f64, W>`. One
//! implementation for every target; LLVM picks the instructions (NEON, AVX2, AVX-512, …) from the
//! enabled target features.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use std::simd::StdFloat;
use std::simd::prelude::*;

use super::{Lanes, exp_poly, ln_poly};
use crate::{AdNum, Real};

macro_rules! pack {
    ($(#[$doc:meta])* $T:ident, $e:ident, $i:ident, $u:ident, $bias:expr, $shift:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq)]
        pub struct $T<const W: usize>(Simd<$e, W>);

        impl<const W: usize> $T<W> {
            #[inline(always)]
            fn splat(k: $e) -> Simd<$e, W> {
                Simd::splat(k)
            }
            /// `a·b + c`: fused where the target has an FMA instruction, else two rounded ops
            /// (`Simd::mul_add` is a libm call per lane there).
            #[inline(always)]
            fn fma(a: Simd<$e, W>, b: Simd<$e, W>, c: Simd<$e, W>) -> Simd<$e, W> {
                #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
                let r = a.mul_add(b, c);
                #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
                let r = a * b + c;
                r
            }
        }

        impl<const W: usize> fmt::Debug for $T<W> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.to_array().fmt(f)
            }
        }
        impl<const W: usize> From<f64> for $T<W> {
            #[inline(always)]
            fn from(k: f64) -> Self {
                Self(Self::splat(k as $e))
            }
        }

        impl<const W: usize> Add for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn add(self, o: Self) -> Self {
                Self(self.0 + o.0)
            }
        }
        impl<const W: usize> Sub for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, o: Self) -> Self {
                Self(self.0 - o.0)
            }
        }
        impl<const W: usize> Mul for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, o: Self) -> Self {
                Self(self.0 * o.0)
            }
        }
        impl<const W: usize> Div for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn div(self, o: Self) -> Self {
                Self(self.0 / o.0)
            }
        }
        impl<const W: usize> Neg for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }
        impl<const W: usize> Add<f64> for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn add(self, k: f64) -> Self {
                Self(self.0 + Self::splat(k as $e))
            }
        }
        impl<const W: usize> Sub<f64> for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, k: f64) -> Self {
                Self(self.0 - Self::splat(k as $e))
            }
        }
        impl<const W: usize> Mul<f64> for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, k: f64) -> Self {
                Self(self.0 * Self::splat(k as $e))
            }
        }
        impl<const W: usize> Div<f64> for $T<W> {
            type Output = Self;
            #[inline(always)]
            fn div(self, k: f64) -> Self {
                Self(self.0 / Self::splat(k as $e))
            }
        }
        impl<const W: usize> AddAssign for $T<W> {
            #[inline(always)]
            fn add_assign(&mut self, o: Self) {
                self.0 += o.0;
            }
        }
        impl<const W: usize> SubAssign for $T<W> {
            #[inline(always)]
            fn sub_assign(&mut self, o: Self) {
                self.0 -= o.0;
            }
        }

        impl<const W: usize> AdNum for $T<W> {
            /// See [`exp_poly`](super::exp_poly).
            #[inline(always)]
            fn exp(self) -> Self {
                use exp_poly::$e::bits::*;
                let x = self.0.simd_clamp(Self::splat(LO), Self::splat(HI));
                let n = (x * Self::splat(LOG2E)).round();
                let r = Self::fma(n, Self::splat(-LN2_HI), x);
                let r = Self::fma(n, Self::splat(-LN2_LO), r);
                let mut p = Self::splat(P[0]);
                for &c in &P[1..] {
                    p = Self::fma(p, r, Self::splat(c));
                }
                let scale = (n.cast::<$i>() + Simd::splat($bias)) << Simd::splat($shift);
                Self(p * Simd::<$e, W>::from_bits(scale.cast::<$u>()))
            }
            /// See [`ln_poly`](super::ln_poly).
            #[inline(always)]
            fn ln(self) -> Self {
                use exp_poly::$e::bits::{LN2_HI, LN2_LO};
                use ln_poly::$e::bits::*;
                let x = self.0;
                let sub = x.simd_lt(Self::splat(<$e>::MIN_POSITIVE));
                let xs = sub.select(x * Self::splat(SCALE), x);
                let ix = xs.to_bits() - Simd::splat(SQRT_HALF);
                let e = (ix.cast::<$i>() >> Simd::splat($shift)).cast::<$e>();
                let e = e - sub.select(Self::splat(SCALE_LOG2), Self::splat(0.0));
                let f = Simd::<$e, W>::from_bits((ix & Simd::splat(MANT)) + Simd::splat(SQRT_HALF)) - Self::splat(1.0);
                let s = f / (f + Self::splat(2.0));
                let z = s * s;
                let mut p = Self::splat(P[0]);
                for &c in &P[1..] {
                    p = Self::fma(p, z, Self::splat(c));
                }
                let hfsq = Self::splat(0.5) * f * f;
                let lo = Self::fma(s, Self::fma(z, p, hfsq), e * Self::splat(LN2_LO));
                let y = Self::fma(e, Self::splat(LN2_HI), lo - hfsq + f);
                let y = x.simd_eq(Self::splat(0.0)).select(Self::splat(<$e>::NEG_INFINITY), y);
                let y = x.simd_eq(Self::splat(<$e>::INFINITY)).select(x, y);
                Self(x.simd_ge(Self::splat(0.0)).select(y, Self::splat(<$e>::NAN)))
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                Self(self.0.sqrt())
            }
            /// [`powf_small`](super::powf_small) where it applies, else `exp(k·ln x)`. As in `std`,
            /// integer `k` takes negative `x` (odd `k` keeps the sign); unlike `std`, a `-inf` base
            /// with non-integer `k` gives NaN. Relative error grows with `|k·ln x|`.
            #[inline(always)]
            fn powf(self, k: f64) -> Self {
                if let Some(r) = super::powf_small(self, k) {
                    return r;
                }
                let k = k as $e;
                if k.fract() != 0.0 {
                    return (self.ln() * k as f64).exp();
                }
                let r = (Self(self.0.abs()).ln() * k as f64).exp();
                if (k * 0.5).fract() != 0.0 { Self(r.0.copysign(self.0)) } else { r }
            }
            /// `max(x, 0)` with NaN propagating (`simd_max` would return 0).
            #[inline(always)]
            fn relu(self) -> Self {
                Self(self.0.simd_lt(Self::splat(0.0)).select(Self::splat(0.0), self.0))
            }
            #[inline(always)]
            fn detach(self) -> Self {
                self
            }
            #[inline(always)]
            fn select_gt(a: Self, b: Self, x: Self, y: Self) -> Self {
                Self(a.0.simd_gt(b.0).select(x.0, y.0))
            }
            #[inline(always)]
            fn rsub(self, k: f64) -> Self {
                Self(Self::splat(k as $e) - self.0)
            }
            #[inline(always)]
            fn rdiv(self, k: f64) -> Self {
                Self(Self::splat(k as $e) / self.0)
            }
        }

        impl<const W: usize> Real for $T<W> {
            type Lanes = Self;
            type T<X: Copy> = [X; 0];
            #[inline(always)]
            fn lift(v: Self) -> Self {
                v
            }
            #[inline(always)]
            fn value(self) -> Self {
                self
            }
            #[inline(always)]
            fn tangents(self) -> [Self; 0] {
                []
            }
            #[inline(always)]
            fn mul_add(self, b: Self, c: Self) -> Self {
                Self(Self::fma(self.0, b.0, c.0))
            }
            #[inline(always)]
            fn nmul_add(self, b: Self, c: Self) -> Self {
                Self(Self::fma(-self.0, b.0, c.0))
            }
            #[inline(always)]
            fn neg_zero() -> Self {
                Self(Self::splat(-0.0))
            }
            #[inline]
            fn per_lane(f: impl Fn(usize) -> Self) -> Self {
                Self(Simd::from_array(std::array::from_fn(|l| f(l).0[l])))
            }
        }

        impl<const W: usize> Lanes for $T<W> {
            type Elem = $e;
            const W: usize = W;
            #[inline(always)]
            fn load(src: &[$e]) -> Self {
                Self(Simd::from_slice(&src[..W]))
            }
            #[inline(always)]
            fn load_f32(src: &[f32]) -> Self {
                Self(Simd::<f32, W>::from_slice(&src[..W]).cast())
            }
            #[inline(always)]
            fn load_f64(src: &[f64]) -> Self {
                Self(Simd::<f64, W>::from_slice(&src[..W]).cast())
            }
            #[inline(always)]
            fn store(self, dst: &mut [$e]) {
                self.0.copy_to_slice(&mut dst[..W]);
            }
            #[inline(always)]
            fn lane(self, i: usize) -> f64 {
                self.0[i] as f64
            }
            #[inline(always)]
            fn from_fn(mut f: impl FnMut(usize) -> f64) -> Self {
                Self(Simd::from_array(std::array::from_fn(|i| f(i) as $e)))
            }
        }
    };
}

// type, element, same-width int and uint, exponent bias - 1, mantissa bits
pack! {
    /// `W` lanes of `f32`.
    F32, f32, i32, u32, 126, 23
}
pack! {
    /// `W` lanes of `f64`.
    F64, f64, i64, u64, 1022, 52
}
