//! `#[kernels]` option models on generic impl blocks, and engine constants that stay usable
//! without the engine traits in scope (only `AdNum` and the attribute are imported).

use raads::{AdNum, kernels};
use std::marker::PhantomData;

/// American put with a softplus exercise of width `OBSTACLE_EPS`, also read by a kernel.
pub struct SmoothPut;

#[kernels(raads_engines::pde1d)]
impl SmoothPut {
    pub const OBSTACLE: bool = true;
    pub const OBSTACLE_EPS: f64 = 0.01;

    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
        let (r, sigma) = (p[1], p[2]);
        [0.5 * sigma * sigma, r - 0.5 * sigma * sigma, r]
    }

    pub fn initial<N: AdNum>(#[data] x: N, p: [N; 5]) -> N {
        (p[4] - x.exp()).smooth_relu(Self::OBSTACLE_EPS)
    }

    pub fn x0<N: AdNum>(p: [N; 5]) -> N {
        p[0].ln()
    }

    pub fn horizon<N: AdNum>(p: [N; 5]) -> N {
        p[3]
    }
}

/// GBM call with the strike as a const generic.
pub struct Call<const K: usize>;

#[kernels(raads_engines::paths)]
impl<const K: usize> Call<K> {
    const STRIKE: f64 = K as f64;

    pub fn setup<N: AdNum>(p: [N; 4], #[data] n_steps: N) -> [N; 2] {
        let dt = p[3] / n_steps;
        [(p[1] - 0.5 * p[2] * p[2]) * dt, p[2] * dt.sqrt()]
    }

    pub fn init<N: AdNum>(p: [N; 4]) -> [N; 1] {
        [p[0].ln()]
    }

    pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] + c[1] * z[0]]
    }

    pub fn functional<N: AdNum>(s: [N; 1], p: [N; 4], _c: [N; 2]) -> N {
        (s[0].exp() - Self::STRIKE).relu() * (-p[1] * p[3]).exp()
    }
}

pub trait Vol {
    const SIGMA: f64;
}

pub struct Low;

impl Vol for Low {
    const SIGMA: f64 = 0.2;
}

/// FD European call with the volatility from a type parameter named `T`, like the GAT
/// parameter of the trait.
pub struct EuroCall<T>(PhantomData<T>);

#[kernels(raads_engines::pde1d)]
impl<T: Vol> EuroCall<T> {
    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 3]) -> [N; 3] {
        let v = T::SIGMA * T::SIGMA;
        [N::from(0.5 * v), p[1] - 0.5 * v, p[1]]
    }

    pub fn initial<N: AdNum>(#[data] x: N, _p: [N; 3]) -> N {
        (x.exp() - 100.0).relu()
    }

    pub fn x0<N: AdNum>(p: [N; 3]) -> N {
        p[0].ln()
    }

    pub fn horizon<N: AdNum>(p: [N; 3]) -> N {
        p[2]
    }
}

#[test]
fn engine_constants_stay_inherent() {
    use raads_engines::pde1d::{self, Grid, PdeModel};
    const { assert!(SmoothPut::OBSTACLE) };
    assert_eq!(SmoothPut::OBSTACLE_EPS, 0.01);
    const { assert!(<SmoothPut as PdeModel>::OBSTACLE) };
    assert_eq!(<SmoothPut as PdeModel>::OBSTACLE_EPS, 0.01);
    const { assert!(!<EuroCall<Low> as PdeModel>::OBSTACLE) };

    let grid = Grid::new(100f64.ln() - 1.2, 100f64.ln() + 1.2, 201, 200);
    let r = pde1d::gradient::<SmoothPut, f64>(&[100.0, 0.05, 0.2, 1.0, 100.0], &grid);
    assert!((r.value - 6.09).abs() < 0.05, "{}", r.value);
}

#[test]
fn generic_models() {
    use raads_engines::paths;
    use raads_engines::pde1d::{self, Grid};
    // Black-Scholes call, S = K = 100, r = 0.05, sigma = 0.2, T = 1
    let grid = Grid::new(100f64.ln() - 1.2, 100f64.ln() + 1.2, 201, 200);
    let r = pde1d::gradient::<EuroCall<Low>, f64>(&[100.0, 0.05, 1.0], &grid);
    assert!((r.value - 10.4506).abs() < 0.01, "{}", r.value);

    let p = [100.0, 0.05, 0.2, 1.0];
    let s = Call::<100>::init(p);
    assert_eq!(Call::<100>::functional([4.0f64.ln()], p, [0.0; 2]), 0.0);
    let n_paths = 4096;
    let normals: Vec<f32> = (0..n_paths).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
    let m = paths::gradient::<Call<100>, f64>(&p, 1, n_paths, &normals);
    let want = Call::<100>::functional(Call::<100>::step(s, Call::<100>::setup(p, 1.0), [0.5]), p, [0.0; 2]) / 2.0;
    assert!((m.value - want).abs() < 1e-12, "{} vs {want}", m.value);
}
