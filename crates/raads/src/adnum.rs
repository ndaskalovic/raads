//! The operations a kernel may use.
//!
//! Kernels are written once, generic over `N: AdNum`. `#[kernel]` turns them into
//! straight-line forward and reverse code; the same source also runs eagerly on any
//! [`Real`](crate::Real) (`f64`, SIMD packs, [`Dual`](crate::Dual)) and on the graph
//! tracer used to test the macro. Comparisons exist only through [`AdNum::select_gt`],
//! so a kernel is branch-free on differentiable values and its graph never changes.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A differentiable number: the operations a kernel may use.
///
/// Sealed: implemented by `f64`, the SIMD packs and [`Dual`](crate::Dual) (and the tracer),
/// so new operations can be added without breaking user code.
pub trait AdNum:
    crate::sealed::Sealed
    + Copy
    + From<f64>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    /// `e^self`
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// `self ** k`
    fn powf(self, k: f64) -> Self;
    /// `max(self, 0)`. NaN propagates on every backend: `relu(NaN) = NaN` (its derivative is 0).
    fn relu(self) -> Self;
    /// Same value, zero derivative (stop-gradient).
    fn detach(self) -> Self;
    /// `x` where `a > b`, else `y`, lane-wise. The derivative flows only into the chosen branch.
    fn select_gt(a: Self, b: Self, x: Self, y: Self) -> Self;
    /// `k - self`
    fn rsub(self, k: f64) -> Self;
    /// `k / self`
    fn rdiv(self, k: f64) -> Self;
    /// `k + self`
    #[inline(always)]
    fn radd(self, k: f64) -> Self {
        self + k
    }
    /// `k * self`
    #[inline(always)]
    fn rmul(self, k: f64) -> Self {
        self * k
    }

    /// `max(self, b)` via [`select_gt`](AdNum::select_gt).
    #[inline(always)]
    fn maximum(self, b: Self) -> Self {
        Self::select_gt(self, b, self, b)
    }
    /// `min(self, b)` via [`select_gt`](AdNum::select_gt).
    #[inline(always)]
    fn minimum(self, b: Self) -> Self {
        Self::select_gt(self, b, b, self)
    }
    /// Softplus `max(x, 0)` of width `eps`, `eps * ln(1 + exp(x / eps))`, overflow-safe.
    /// Gives consistent pathwise second derivatives; bias O(eps²).
    #[inline(always)]
    fn smooth_relu(self, eps: f64) -> Self {
        let ax = Self::select_gt(self, Self::from(0.0), self, -self);
        self.relu() + (-ax / eps).exp().radd(1.0).ln().rmul(eps)
    }
}
