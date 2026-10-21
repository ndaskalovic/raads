//! Black-Scholes European call and American puts for the 1-D PDE engine, in x = log S (or
//! log(S/K)) and time to maturity: `initial` is the payoff, `x0` today's log-spot, `horizon`
//! the maturity, and early exercise is the obstacle (`OBSTACLE`: V = max(V, payoff) after
//! every step). The engine's value is the price and its gradient the Greeks.

use raads::{AdNum, kernels};

/// params: s0, r, sigma, T, K.
pub struct BsEuroCall;

#[kernels(raads_engines::pde1d)]
impl BsEuroCall {
    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
        let r = p[1];
        let sigma = p[2];
        [0.5 * sigma * sigma, r - 0.5 * sigma * sigma, r]
    }

    /// Payoff.
    pub fn initial<N: AdNum>(#[data] x: N, p: [N; 5]) -> N {
        (x.exp() - p[4]).relu()
    }

    /// Log-spot.
    pub fn x0<N: AdNum>(p: [N; 5]) -> N {
        p[0].ln()
    }

    /// Maturity.
    pub fn horizon<N: AdNum>(p: [N; 5]) -> N {
        p[3]
    }
}

/// American put. params: s0, r, sigma, T, K.
pub struct BsAmericanPut;

#[kernels(raads_engines::pde1d)]
impl BsAmericanPut {
    /// Early exercise.
    const OBSTACLE: bool = true;

    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
        let r = p[1];
        let sigma = p[2];
        [0.5 * sigma * sigma, r - 0.5 * sigma * sigma, r]
    }

    /// Payoff (also the exercise value).
    pub fn initial<N: AdNum>(#[data] x: N, p: [N; 5]) -> N {
        (p[4] - x.exp()).relu()
    }

    /// Log-spot.
    pub fn x0<N: AdNum>(p: [N; 5]) -> N {
        p[0].ln()
    }

    /// Maturity.
    pub fn horizon<N: AdNum>(p: [N; 5]) -> N {
        p[3]
    }
}

/// American put with a softplus-smoothed exercise decision (width 0.002), so second-order
/// Greeks w.r.t. boundary-moving parameters are well defined.
pub struct BsAmericanPutSmooth;

#[kernels(raads_engines::pde1d)]
impl BsAmericanPutSmooth {
    /// Early exercise.
    const OBSTACLE: bool = true;
    /// Width of the smoothed exercise max.
    const OBSTACLE_EPS: f64 = 0.002;

    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
        let r = p[1];
        let sigma = p[2];
        [0.5 * sigma * sigma, r - 0.5 * sigma * sigma, r]
    }

    /// Payoff (also the exercise value).
    pub fn initial<N: AdNum>(#[data] x: N, p: [N; 5]) -> N {
        (p[4] - x.exp()).relu()
    }

    /// Log-spot.
    pub fn x0<N: AdNum>(p: [N; 5]) -> N {
        p[0].ln()
    }

    /// Maturity.
    pub fn horizon<N: AdNum>(p: [N; 5]) -> N {
        p[3]
    }
}

/// Unit-strike put on x = log(S/K) for [`option`](crate::option): V(S, K, T) = K u(log(S/K), T),
/// so one solve prices every strike. params: r, q (continuous dividend yield), sigma.
pub struct BsPutMoneyness<const AMERICAN: bool>;

#[kernels(raads_engines::pde1d)]
impl<const A: bool> BsPutMoneyness<A> {
    /// Early exercise.
    const OBSTACLE: bool = A;

    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 3]) -> [N; 3] {
        let r = p[0];
        let q = p[1];
        let sigma = p[2];
        [0.5 * sigma * sigma, r - q - 0.5 * sigma * sigma, r]
    }

    /// Unit-strike payoff.
    pub fn initial<N: AdNum>(#[data] x: N, _p: [N; 3]) -> N {
        x.exp().rsub(1.0).relu()
    }

    /// Unused by `snapshots`.
    pub fn x0<N: AdNum>(_p: [N; 3]) -> N {
        N::from(0.0)
    }

    /// Unused: maturities are the snapshot horizons.
    pub fn horizon<N: AdNum>(_p: [N; 3]) -> N {
        N::from(0.0)
    }
}
