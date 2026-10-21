//! GBM European and Asian calls (log-space exact scheme) for the path engine. The engine's
//! `functional` is the discounted payoff, so its value is the price and its gradient the Greeks.

use raads::{AdNum, kernels};

/// European call. params: s0, r, sigma, T, K. coeffs: drift, vol·√dt. state: log S.
pub struct GbmCall;

#[kernels(raads_engines::paths)]
impl GbmCall {
    pub fn setup<N: AdNum>(p: [N; 5], #[data] n_steps: N) -> [N; 2] {
        let r = p[1];
        let sigma = p[2];
        let dt = p[3] / n_steps;
        [(r - 0.5 * sigma * sigma) * dt, sigma * dt.sqrt()]
    }

    pub fn init<N: AdNum>(p: [N; 5]) -> [N; 1] {
        [p[0].ln()]
    }

    pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] + c[1] * z[0]]
    }

    /// Discounted payoff.
    pub fn functional<N: AdNum>(s: [N; 1], p: [N; 5], _c: [N; 2]) -> N {
        (s[0].exp() - p[4]).relu() * (-p[1] * p[3]).exp()
    }
}

/// Arithmetic-average Asian call. state: log S, running sum of S. coeffs: drift, vol·√dt, 1/n.
pub struct GbmAsianCall;

#[kernels(raads_engines::paths)]
impl GbmAsianCall {
    pub fn setup<N: AdNum>(p: [N; 5], #[data] n_steps: N) -> [N; 3] {
        let r = p[1];
        let sigma = p[2];
        let dt = p[3] / n_steps;
        [(r - 0.5 * sigma * sigma) * dt, sigma * dt.sqrt(), 1.0 / n_steps]
    }

    pub fn init<N: AdNum>(p: [N; 5]) -> [N; 2] {
        [p[0].ln(), N::from(0.0)]
    }

    pub fn step<N: AdNum>(s: [N; 2], c: [N; 3], #[data] z: [N; 1]) -> [N; 2] {
        let x = s[0] + c[0] + c[1] * z[0];
        [x, s[1] + x.exp()]
    }

    /// Discounted payoff.
    pub fn functional<N: AdNum>(s: [N; 2], p: [N; 5], c: [N; 3]) -> N {
        (s[1] * c[2] - p[4]).relu() * (-p[1] * p[3]).exp()
    }
}

/// European call with a softplus-smoothed payoff of width `EPS`, for pathwise second-order Greeks.
pub struct GbmCallSmooth;

#[kernels(raads_engines::paths)]
impl GbmCallSmooth {
    pub const EPS: f64 = 0.5;

    pub fn setup<N: AdNum>(p: [N; 5], #[data] n_steps: N) -> [N; 2] {
        let r = p[1];
        let sigma = p[2];
        let dt = p[3] / n_steps;
        [(r - 0.5 * sigma * sigma) * dt, sigma * dt.sqrt()]
    }

    pub fn init<N: AdNum>(p: [N; 5]) -> [N; 1] {
        [p[0].ln()]
    }

    pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] + c[1] * z[0]]
    }

    /// Discounted smoothed payoff.
    pub fn functional<N: AdNum>(s: [N; 1], p: [N; 5], _c: [N; 2]) -> N {
        (s[0].exp() - p[4]).smooth_relu(Self::EPS) * (-p[1] * p[3]).exp()
    }
}
