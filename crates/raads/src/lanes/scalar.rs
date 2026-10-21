//! `f64` as a one-lane pack (e.g. a single FD problem).

use super::Lanes;
use crate::{AdNum, Real};

impl crate::sealed::Sealed for f64 {}

impl AdNum for f64 {
    #[inline(always)]
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    #[inline(always)]
    fn ln(self) -> f64 {
        f64::ln(self)
    }
    #[inline(always)]
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn powf(self, k: f64) -> f64 {
        super::powf_small(self, k).unwrap_or_else(|| f64::powf(self, k))
    }
    #[inline(always)]
    fn relu(self) -> f64 {
        if self < 0.0 { 0.0 } else { self }
    }
    #[inline(always)]
    fn detach(self) -> f64 {
        self
    }
    #[inline(always)]
    fn select_gt(a: f64, b: f64, x: f64, y: f64) -> f64 {
        if a > b { x } else { y }
    }
    #[inline(always)]
    fn rsub(self, k: f64) -> f64 {
        k - self
    }
    #[inline(always)]
    fn rdiv(self, k: f64) -> f64 {
        k / self
    }
}

impl Real for f64 {
    type Lanes = f64;
    #[inline(always)]
    fn lift(v: f64) -> f64 {
        v
    }
    #[inline(always)]
    fn value(self) -> f64 {
        self
    }
    type T<X: Copy> = [X; 0];
    #[inline(always)]
    fn tangents(self) -> [f64; 0] {
        []
    }
    #[inline(always)]
    fn mul_add(self, b: f64, c: f64) -> f64 {
        mul_add!(self, b, c)
    }
    #[inline(always)]
    fn nmul_add(self, b: f64, c: f64) -> f64 {
        mul_add!(-self, b, c)
    }
    #[inline(always)]
    fn neg_zero() -> f64 {
        -0.0
    }
    #[inline]
    fn per_lane(f: impl Fn(usize) -> f64) -> f64 {
        f(0)
    }
}

impl Lanes for f64 {
    type Elem = f64;
    const W: usize = 1;
    #[inline(always)]
    fn load(src: &[f64]) -> f64 {
        src[0]
    }
    #[inline(always)]
    fn load_f32(src: &[f32]) -> f64 {
        src[0] as f64
    }
    #[inline(always)]
    fn load_f64(src: &[f64]) -> f64 {
        src[0]
    }
    #[inline(always)]
    fn store(self, dst: &mut [f64]) {
        dst[0] = self;
    }
    #[inline(always)]
    fn lane(self, _i: usize) -> f64 {
        self
    }
    #[inline(always)]
    fn from_fn(mut f: impl FnMut(usize) -> f64) -> f64 {
        f(0)
    }
}
