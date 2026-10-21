//! Option surfaces: for a unit-strike model on x = log(S/K) (e.g.
//! [`BsPutMoneyness`](crate::bs_fd::BsPutMoneyness)), `V(S, K, T) = K u(log(S/K), T)`, so one
//! [`pde1d::snapshots`](raads_engines::pde1d::snapshots) solve over the maturities prices
//! every strike and maturity. With [`Dual`](raads::Dual) parameters each lane carries one
//! forward tangent (e.g. vega in one lane, rho in another);
//! [`pde1d::snapshots_weighted`](raads_engines::pde1d::snapshots_weighted) with weights
//! `position · K` gives a book's value and its gradient in one reverse sweep.

use raads::{AdNum, Real};
use raads_engines::pde1d::Snapshots;

/// One option read off a unit-strike surface by [`option`]: price and grid Greeks.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SurfaceOption<R> {
    /// Price.
    pub price: R,
    /// dV/dS.
    pub delta: R,
    /// d²V/dS².
    pub gamma: R,
    /// dV/dt in calendar time (= -dV/dT).
    pub theta: R,
    /// dV/dK.
    pub dvdk: R,
}

/// Price and grid Greeks of the option with this spot and strike at maturity index `m` of a
/// unit-strike surface. Theta comes from the PDE (`u_τ = L u` on continuation nodes): no dt
/// error. Reads as [`Snapshots::read`]; panics if `m` is out of range.
pub fn option<R: Real>(s: &Snapshots<R>, m: usize, spot: R::Lanes, strike: R::Lanes) -> SurfaceOption<R> {
    let x = (spot / strike).ln();
    let r = s.read(m, x);
    let rt = s.read_tau(m, x);
    let (k, s) = (R::lift(strike), R::lift(spot));
    SurfaceOption { price: k * r[0], delta: k * r[1] / s, gamma: k * (r[2] - r[1]) / (s * s), theta: -(k * rt[0]), dvdk: r[0] - r[1] }
}
