//! Runtime value types for compiled kernels and engines.
//!
//! [`Real`] is what generated code and engines compute with. Every [`Lanes`] pack is a
//! `Real` (first order); [`Dual`] adds one tangent per lane, so running a reverse sweep
//! on `Dual` values gives forward-over-reverse: Hessian-vector products.

use crate::lanes::Lanes;
use crate::{AdNum, Arr};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// A runtime value of generated code and engines: a [`Lanes`] pack or a [`Dual`] of one.
/// Sealed through [`AdNum`].
pub trait Real: AdNum + AddAssign + SubAssign + Send + Sync + 'static {
    /// The SIMD pack holding values (and tangents).
    type Lanes: Lanes;
    /// Constant (zero tangent).
    #[inline(always)]
    fn splat(k: f64) -> Self {
        Self::from(k)
    }
    /// Constant pack (zero tangent).
    fn lift(v: Self::Lanes) -> Self;
    /// The values.
    fn value(self) -> Self::Lanes;
    /// `[X; number of tangent directions]`: 0 for a [`Lanes`] pack, 1 for [`Dual`], `K` for
    /// [`DualVec<_, K>`](DualVec).
    type T<X: Copy>: Arr<X>;
    /// The tangents, one pack per direction.
    fn tangents(self) -> Self::T<Self::Lanes>;
    /// `self * b + c`, fused.
    fn mul_add(self, b: Self, c: Self) -> Self;
    /// `c - self * b`, fused.
    fn nmul_add(self, b: Self, c: Self) -> Self;
    /// `self * k + c` for a constant `k`, fused (no work on `k`'s zero tangent).
    #[inline(always)]
    fn mul_add_k(self, k: f64, c: Self) -> Self {
        self.mul_add(Self::splat(k), c)
    }
    /// `c - self * k` for a constant `k`, fused.
    #[inline(always)]
    fn nmul_add_k(self, k: f64, c: Self) -> Self {
        self.nmul_add(Self::splat(k), c)
    }
    /// `self + l` for a passive pack `l` (no tangent): generated code mixes active values with
    /// values that depend only on data and constants through these, so a [`Dual`] never computes
    /// with a zero tangent.
    #[inline(always)]
    fn add_l(self, l: Self::Lanes) -> Self {
        self + Self::lift(l)
    }
    /// `self - l` for a passive pack `l`.
    #[inline(always)]
    fn sub_l(self, l: Self::Lanes) -> Self {
        self - Self::lift(l)
    }
    /// `l - self` for a passive pack `l`.
    #[inline(always)]
    fn l_sub(self, l: Self::Lanes) -> Self {
        Self::lift(l) - self
    }
    /// `self * l` for a passive pack `l`.
    #[inline(always)]
    fn mul_l(self, l: Self::Lanes) -> Self {
        self * Self::lift(l)
    }
    /// `self / l` for a passive pack `l`.
    #[inline(always)]
    fn div_l(self, l: Self::Lanes) -> Self {
        self / Self::lift(l)
    }
    /// `l / self` for a passive pack `l`.
    #[inline(always)]
    fn l_div(self, l: Self::Lanes) -> Self {
        Self::lift(l) / self
    }
    /// `self * l + c` for a passive pack `l`, fused.
    #[inline(always)]
    fn mul_add_l(self, l: Self::Lanes, c: Self) -> Self {
        self.mul_add(Self::lift(l), c)
    }
    /// `c - self * l` for a passive pack `l`, fused.
    #[inline(always)]
    fn nmul_add_l(self, l: Self::Lanes, c: Self) -> Self {
        self.nmul_add(Self::lift(l), c)
    }
    /// Additive identity with every component `-0.0`, so `acc + x` folds to `x`.
    fn neg_zero() -> Self;
    /// Lane `l` of the result is lane `l` of `f(l)` (per-lane gathers, e.g. interpolation).
    fn per_lane(f: impl Fn(usize) -> Self) -> Self;
}

/// Value and one tangent direction, lane-wise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<L> {
    /// Value.
    pub v: L,
    /// Tangent.
    pub t: L,
}

impl<L> crate::sealed::Sealed for Dual<L> {}

impl<L: Lanes> Dual<L> {
    /// Value `v` with tangent `t`.
    #[inline(always)]
    pub fn new(v: L, t: L) -> Self {
        Dual { v, t }
    }
}

impl<L: Lanes> From<f64> for Dual<L> {
    #[inline(always)]
    fn from(k: f64) -> Self {
        Dual { v: L::from(k), t: L::from(0.0) }
    }
}

impl<L: Lanes> Add for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self {
        Dual { v: self.v + r.v, t: self.t + r.t }
    }
}
impl<L: Lanes> Sub for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self {
        Dual { v: self.v - r.v, t: self.t - r.t }
    }
}
impl<L: Lanes> Mul for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self {
        Dual { v: self.v * r.v, t: self.v.mul_add(r.t, self.t * r.v) }
    }
}
impl<L: Lanes> Div for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn div(self, r: Self) -> Self {
        let q = self.v / r.v;
        Dual { v: q, t: q.nmul_add(r.t, self.t) / r.v }
    }
}
impl<L: Lanes> Neg for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Dual { v: -self.v, t: -self.t }
    }
}
impl<L: Lanes> Add<f64> for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn add(self, k: f64) -> Self {
        Dual { v: self.v + k, t: self.t }
    }
}
impl<L: Lanes> Sub<f64> for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, k: f64) -> Self {
        Dual { v: self.v - k, t: self.t }
    }
}
impl<L: Lanes> Mul<f64> for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, k: f64) -> Self {
        Dual { v: self.v * k, t: self.t * k }
    }
}
impl<L: Lanes> Div<f64> for Dual<L> {
    type Output = Self;
    #[inline(always)]
    fn div(self, k: f64) -> Self {
        Dual { v: self.v / k, t: self.t / k }
    }
}
impl<L: Lanes> AddAssign for Dual<L> {
    #[inline(always)]
    fn add_assign(&mut self, r: Self) {
        *self = *self + r;
    }
}
impl<L: Lanes> SubAssign for Dual<L> {
    #[inline(always)]
    fn sub_assign(&mut self, r: Self) {
        *self = *self - r;
    }
}

impl<L: Lanes> AdNum for Dual<L> {
    #[inline(always)]
    fn exp(self) -> Self {
        let e = self.v.exp();
        Dual { v: e, t: e * self.t }
    }
    #[inline(always)]
    fn ln(self) -> Self {
        Dual { v: self.v.ln(), t: self.t / self.v }
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        Dual { v: s, t: self.t * 0.5 / s }
    }
    #[inline(always)]
    fn powf(self, k: f64) -> Self {
        Dual { v: self.v.powf(k), t: self.v.powf(k - 1.0) * k * self.t }
    }
    #[inline(always)]
    fn relu(self) -> Self {
        let z = L::from(0.0);
        Dual { v: self.v.relu(), t: L::select_gt(self.v, z, self.t, z) }
    }
    #[inline(always)]
    fn detach(self) -> Self {
        Dual { v: self.v, t: L::from(0.0) }
    }
    #[inline(always)]
    fn select_gt(a: Self, b: Self, x: Self, y: Self) -> Self {
        Dual { v: L::select_gt(a.v, b.v, x.v, y.v), t: L::select_gt(a.v, b.v, x.t, y.t) }
    }
    #[inline(always)]
    fn rsub(self, k: f64) -> Self {
        Dual { v: self.v.rsub(k), t: -self.t }
    }
    #[inline(always)]
    fn rdiv(self, k: f64) -> Self {
        let q = self.v.rdiv(k);
        Dual { v: q, t: -(q * self.t) / self.v }
    }
}

impl<L: Lanes> Real for Dual<L> {
    type Lanes = L;
    #[inline(always)]
    fn lift(v: L) -> Self {
        Dual { v, t: L::from(0.0) }
    }
    #[inline(always)]
    fn value(self) -> L {
        self.v
    }
    type T<X: Copy> = [X; 1];
    #[inline(always)]
    fn tangents(self) -> [L; 1] {
        [self.t]
    }
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        Dual { v: self.v.mul_add(b.v, c.v), t: self.v.mul_add(b.t, self.t.mul_add(b.v, c.t)) }
    }
    #[inline(always)]
    fn nmul_add(self, b: Self, c: Self) -> Self {
        Dual { v: self.v.nmul_add(b.v, c.v), t: self.v.nmul_add(b.t, self.t.nmul_add(b.v, c.t)) }
    }
    #[inline(always)]
    fn mul_add_k(self, k: f64, c: Self) -> Self {
        Dual { v: self.v.mul_add_k(k, c.v), t: self.t.mul_add_k(k, c.t) }
    }
    #[inline(always)]
    fn add_l(self, l: L) -> Self {
        Dual { v: self.v + l, t: self.t }
    }
    #[inline(always)]
    fn sub_l(self, l: L) -> Self {
        Dual { v: self.v - l, t: self.t }
    }
    #[inline(always)]
    fn l_sub(self, l: L) -> Self {
        Dual { v: l - self.v, t: -self.t }
    }
    #[inline(always)]
    fn mul_l(self, l: L) -> Self {
        Dual { v: self.v * l, t: self.t * l }
    }
    #[inline(always)]
    fn div_l(self, l: L) -> Self {
        Dual { v: self.v / l, t: self.t / l }
    }
    #[inline(always)]
    fn l_div(self, l: L) -> Self {
        let q = l / self.v;
        Dual { v: q, t: -(q * self.t) / self.v }
    }
    #[inline(always)]
    fn mul_add_l(self, l: L, c: Self) -> Self {
        Dual { v: self.v.mul_add(l, c.v), t: self.t.mul_add(l, c.t) }
    }
    #[inline(always)]
    fn nmul_add_l(self, l: L, c: Self) -> Self {
        Dual { v: self.v.nmul_add(l, c.v), t: self.t.nmul_add(l, c.t) }
    }
    #[inline(always)]
    fn nmul_add_k(self, k: f64, c: Self) -> Self {
        Dual { v: self.v.nmul_add_k(k, c.v), t: self.t.nmul_add_k(k, c.t) }
    }
    #[inline(always)]
    fn neg_zero() -> Self {
        Dual { v: L::neg_zero(), t: L::neg_zero() }
    }
    #[inline]
    fn per_lane(f: impl Fn(usize) -> Self) -> Self {
        Dual { v: L::from_fn(|l| f(l).v.lane(l)), t: L::from_fn(|l| f(l).t.lane(l)) }
    }
}

/// Value and `K` tangent directions, lane-wise (vector forward mode). Seeding tangent `k` of
/// the inputs with direction `e_k`, one reverse sweep gives `K` Hessian columns (`H e_k` in
/// the adjoints' tangents) instead of `K` sweeps on [`Dual`]. Derivatives that divide by the
/// same value share one reciprocal, so tangents may differ from [`Dual`]'s in the last bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualVec<L, const K: usize> {
    /// Value.
    pub v: L,
    /// Tangents.
    pub t: [L; K],
}

impl<L, const K: usize> crate::sealed::Sealed for DualVec<L, K> {}

impl<L: Lanes, const K: usize> DualVec<L, K> {
    /// Value `v` with tangents `t`.
    #[inline(always)]
    pub fn new(v: L, t: [L; K]) -> Self {
        DualVec { v, t }
    }
    /// Value `v` with tangent `k` one and the others zero: input `k` of a Hessian.
    #[inline(always)]
    pub fn unit(v: L, k: usize) -> Self {
        DualVec { v, t: std::array::from_fn(|j| L::from(if j == k { 1.0 } else { 0.0 })) }
    }
    /// Value `v`, tangents `d · t_k` (chain rule through a unary function with derivative `d`).
    #[inline(always)]
    fn chain(v: L, d: L, t: [L; K]) -> Self {
        DualVec { v, t: std::array::from_fn(|k| d * t[k]) }
    }
}

impl<L: Lanes, const K: usize> From<f64> for DualVec<L, K> {
    #[inline(always)]
    fn from(k: f64) -> Self {
        DualVec { v: L::from(k), t: [L::from(0.0); K] }
    }
}

impl<L: Lanes, const K: usize> Add for DualVec<L, K> {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self {
        DualVec { v: self.v + r.v, t: std::array::from_fn(|k| self.t[k] + r.t[k]) }
    }
}
impl<L: Lanes, const K: usize> Sub for DualVec<L, K> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self {
        DualVec { v: self.v - r.v, t: std::array::from_fn(|k| self.t[k] - r.t[k]) }
    }
}
impl<L: Lanes, const K: usize> Mul for DualVec<L, K> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self {
        DualVec { v: self.v * r.v, t: std::array::from_fn(|k| self.v.mul_add(r.t[k], self.t[k] * r.v)) }
    }
}
impl<L: Lanes, const K: usize> Div for DualVec<L, K> {
    type Output = Self;
    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)] // one reciprocal shared by the K tangents
    fn div(self, r: Self) -> Self {
        let q = self.v / r.v;
        let inv = r.v.rdiv(1.0);
        DualVec { v: q, t: std::array::from_fn(|k| q.nmul_add(r.t[k], self.t[k]) * inv) }
    }
}
impl<L: Lanes, const K: usize> Neg for DualVec<L, K> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        DualVec { v: -self.v, t: std::array::from_fn(|k| -self.t[k]) }
    }
}
impl<L: Lanes, const K: usize> Add<f64> for DualVec<L, K> {
    type Output = Self;
    #[inline(always)]
    fn add(self, k: f64) -> Self {
        DualVec { v: self.v + k, t: self.t }
    }
}
impl<L: Lanes, const K: usize> Sub<f64> for DualVec<L, K> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, k: f64) -> Self {
        DualVec { v: self.v - k, t: self.t }
    }
}
impl<L: Lanes, const K: usize> Mul<f64> for DualVec<L, K> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, c: f64) -> Self {
        DualVec { v: self.v * c, t: std::array::from_fn(|k| self.t[k] * c) }
    }
}
impl<L: Lanes, const K: usize> Div<f64> for DualVec<L, K> {
    type Output = Self;
    #[inline(always)]
    fn div(self, c: f64) -> Self {
        DualVec { v: self.v / c, t: std::array::from_fn(|k| self.t[k] / c) }
    }
}
impl<L: Lanes, const K: usize> AddAssign for DualVec<L, K> {
    #[inline(always)]
    fn add_assign(&mut self, r: Self) {
        *self = *self + r;
    }
}
impl<L: Lanes, const K: usize> SubAssign for DualVec<L, K> {
    #[inline(always)]
    fn sub_assign(&mut self, r: Self) {
        *self = *self - r;
    }
}

impl<L: Lanes, const K: usize> AdNum for DualVec<L, K> {
    #[inline(always)]
    fn exp(self) -> Self {
        let e = self.v.exp();
        Self::chain(e, e, self.t)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        Self::chain(self.v.ln(), self.v.rdiv(1.0), self.t)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        Self::chain(s, s.rdiv(0.5), self.t)
    }
    #[inline(always)]
    fn powf(self, k: f64) -> Self {
        Self::chain(self.v.powf(k), self.v.powf(k - 1.0) * k, self.t)
    }
    #[inline(always)]
    fn relu(self) -> Self {
        let z = L::from(0.0);
        DualVec { v: self.v.relu(), t: std::array::from_fn(|k| L::select_gt(self.v, z, self.t[k], z)) }
    }
    #[inline(always)]
    fn detach(self) -> Self {
        DualVec { v: self.v, t: [L::from(0.0); K] }
    }
    #[inline(always)]
    fn select_gt(a: Self, b: Self, x: Self, y: Self) -> Self {
        DualVec { v: L::select_gt(a.v, b.v, x.v, y.v), t: std::array::from_fn(|k| L::select_gt(a.v, b.v, x.t[k], y.t[k])) }
    }
    #[inline(always)]
    fn rsub(self, c: f64) -> Self {
        DualVec { v: self.v.rsub(c), t: std::array::from_fn(|k| -self.t[k]) }
    }
    #[inline(always)]
    fn rdiv(self, c: f64) -> Self {
        let q = self.v.rdiv(c);
        Self::chain(q, -(q / self.v), self.t)
    }
}

impl<L: Lanes, const K: usize> Real for DualVec<L, K> {
    type Lanes = L;
    type T<X: Copy> = [X; K];
    #[inline(always)]
    fn lift(v: L) -> Self {
        DualVec { v, t: [L::from(0.0); K] }
    }
    #[inline(always)]
    fn value(self) -> L {
        self.v
    }
    #[inline(always)]
    fn tangents(self) -> [L; K] {
        self.t
    }
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        DualVec { v: self.v.mul_add(b.v, c.v), t: std::array::from_fn(|k| self.v.mul_add(b.t[k], self.t[k].mul_add(b.v, c.t[k]))) }
    }
    #[inline(always)]
    fn nmul_add(self, b: Self, c: Self) -> Self {
        DualVec { v: self.v.nmul_add(b.v, c.v), t: std::array::from_fn(|k| self.v.nmul_add(b.t[k], self.t[k].nmul_add(b.v, c.t[k]))) }
    }
    #[inline(always)]
    fn mul_add_k(self, c: f64, a: Self) -> Self {
        DualVec { v: self.v.mul_add_k(c, a.v), t: std::array::from_fn(|k| self.t[k].mul_add_k(c, a.t[k])) }
    }
    #[inline(always)]
    fn add_l(self, l: L) -> Self {
        DualVec { v: self.v + l, t: self.t }
    }
    #[inline(always)]
    fn sub_l(self, l: L) -> Self {
        DualVec { v: self.v - l, t: self.t }
    }
    #[inline(always)]
    fn l_sub(self, l: L) -> Self {
        DualVec { v: l - self.v, t: std::array::from_fn(|k| -self.t[k]) }
    }
    #[inline(always)]
    fn mul_l(self, l: L) -> Self {
        Self::chain(self.v * l, l, self.t)
    }
    #[inline(always)]
    fn div_l(self, l: L) -> Self {
        Self::chain(self.v / l, l.rdiv(1.0), self.t)
    }
    #[inline(always)]
    fn l_div(self, l: L) -> Self {
        let q = l / self.v;
        Self::chain(q, -(q / self.v), self.t)
    }
    #[inline(always)]
    fn mul_add_l(self, l: L, a: Self) -> Self {
        DualVec { v: self.v.mul_add(l, a.v), t: std::array::from_fn(|k| self.t[k].mul_add(l, a.t[k])) }
    }
    #[inline(always)]
    fn nmul_add_l(self, l: L, a: Self) -> Self {
        DualVec { v: self.v.nmul_add(l, a.v), t: std::array::from_fn(|k| self.t[k].nmul_add(l, a.t[k])) }
    }
    #[inline(always)]
    fn nmul_add_k(self, c: f64, a: Self) -> Self {
        DualVec { v: self.v.nmul_add_k(c, a.v), t: std::array::from_fn(|k| self.t[k].nmul_add_k(c, a.t[k])) }
    }
    #[inline(always)]
    fn neg_zero() -> Self {
        DualVec { v: L::neg_zero(), t: [L::neg_zero(); K] }
    }
    #[inline]
    fn per_lane(f: impl Fn(usize) -> Self) -> Self {
        DualVec { v: L::from_fn(|l| f(l).v.lane(l)), t: std::array::from_fn(|k| L::from_fn(|l| f(l).t[k].lane(l))) }
    }
}
