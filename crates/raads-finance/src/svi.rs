//! European call under GBM with the volatility from a raw-SVI slice.

use raads::{AdNum, kernels};

/// params: s0, r, T, K, a, b, rho, m, sigma. vol = √(w(K) / T) with T detached, so Theta
/// holds the SVI vol fixed.
pub struct SviCall;

#[kernels(raads_engines::paths)]
impl SviCall {
    pub fn setup<N: AdNum>(p: [N; 9], #[data] n_steps: N) -> [N; 2] {
        let r = p[1];
        let t = p[2];
        let km = p[3] - p[7];
        let w = p[4] + p[5] * (p[6] * km + (km * km + p[8] * p[8]).powf(0.5));
        let vol = (w / t.detach()).sqrt();
        let dt = t / n_steps;
        [(r - 0.5 * vol * vol) * dt, vol * dt.sqrt()]
    }

    pub fn init<N: AdNum>(p: [N; 9]) -> [N; 1] {
        [p[0].ln()]
    }

    pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] + c[1] * z[0]]
    }

    /// Discounted payoff.
    pub fn functional<N: AdNum>(s: [N; 1], p: [N; 9], _c: [N; 2]) -> N {
        (s[0].exp() - p[3]).relu() * (-p[1] * p[2]).exp()
    }
}
