//! Generated items are documented.
#![deny(missing_docs)]

use raads::{AdNum, kernels};

/// A path model.
pub struct Walk;

#[kernels(raads_engines::paths)]
impl Walk {
    /// Coefficients.
    pub fn setup<N: AdNum>(p: [N; 2], #[data] n_steps: N) -> [N; 1] {
        [p[1] * (1.0 / n_steps).sqrt()]
    }

    /// Initial state.
    pub fn init<N: AdNum>(p: [N; 2]) -> [N; 1] {
        [p[0]]
    }

    /// One step.
    pub fn step<N: AdNum>(s: [N; 1], c: [N; 1], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] * z[0]]
    }

    /// Value.
    pub fn functional<N: AdNum>(s: [N; 1], _p: [N; 2], _c: [N; 1]) -> N {
        s[0].relu()
    }
}

/// A PDE model.
pub struct Heat;

#[kernels(raads_engines::pde1d)]
impl Heat {
    /// Obstacle.
    pub const OBSTACLE: bool = true;

    /// Coefficients.
    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 2]) -> [N; 3] {
        [p[0], N::from(0.0), N::from(0.0)]
    }

    /// Initial condition.
    pub fn initial<N: AdNum>(#[data] x: N, _p: [N; 2]) -> N {
        (-(x * x)).exp()
    }

    /// Read point.
    pub fn x0<N: AdNum>(_p: [N; 2]) -> N {
        N::from(0.0)
    }

    /// Horizon.
    pub fn horizon<N: AdNum>(p: [N; 2]) -> N {
        p[1]
    }
}

fn main() {}
