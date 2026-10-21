//! `#[kernels]` on generic impl blocks (const generics, a type parameter named like the GAT
//! parameter, a where clause), and engine constants that stay usable without the engine
//! traits in scope (only `AdNum` and the attribute are imported).

use raads::{AdNum, kernels};
use std::marker::PhantomData;

/// Heat equation with a softplus obstacle of width `OBSTACLE_EPS`, also read by a kernel.
/// params: k, x0, T.
pub struct SmoothObstacle;

#[kernels(raads_engines::pde1d)]
impl SmoothObstacle {
    pub const OBSTACLE: bool = true;
    pub const OBSTACLE_EPS: f64 = 0.01;

    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 3]) -> [N; 3] {
        [p[0], N::from(0.0), N::from(0.1)]
    }

    pub fn initial<N: AdNum>(#[data] x: N, _p: [N; 3]) -> N {
        (x * x).rsub(1.0).smooth_relu(Self::OBSTACLE_EPS)
    }

    pub fn x0<N: AdNum>(p: [N; 3]) -> N {
        p[1]
    }

    pub fn horizon<N: AdNum>(p: [N; 3]) -> N {
        p[2]
    }
}

/// Brownian motion from x0 with the functional `max(X_T - K, 0)`, K a const generic.
pub struct Hinge<const K: usize>;

#[kernels(raads_engines::paths)]
impl<const K: usize> Hinge<K> {
    const LEVEL: f64 = K as f64;

    pub fn setup<N: AdNum>(p: [N; 3], #[data] n_steps: N) -> [N; 1] {
        [p[1] * (p[2] / n_steps).sqrt()]
    }

    pub fn init<N: AdNum>(p: [N; 3]) -> [N; 1] {
        [p[0]]
    }

    pub fn step<N: AdNum>(s: [N; 1], c: [N; 1], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] * z[0]]
    }

    pub fn functional<N: AdNum>(s: [N; 1], _p: [N; 3], _c: [N; 1]) -> N {
        (s[0] - Self::LEVEL).relu()
    }
}

pub trait Diffusivity {
    const K: f64;
}

pub struct Half;

impl Diffusivity for Half {
    const K: f64 = 0.5;
}

/// Heat equation with the diffusivity from a type parameter named `T`, like the GAT
/// parameter of the trait. params: x0, T.
pub struct Heat<T>(PhantomData<T>);

#[kernels(raads_engines::pde1d)]
impl<T> Heat<T>
where
    T: Diffusivity,
{
    pub fn coeffs<N: AdNum>(#[data] _x: N, _p: [N; 2]) -> [N; 3] {
        [N::from(T::K), N::from(0.0), N::from(0.0)]
    }

    pub fn initial<N: AdNum>(#[data] x: N, _p: [N; 2]) -> N {
        (-(x * x)).exp()
    }

    pub fn x0<N: AdNum>(p: [N; 2]) -> N {
        p[0]
    }

    pub fn horizon<N: AdNum>(p: [N; 2]) -> N {
        p[1]
    }
}

#[test]
fn engine_constants_stay_inherent() {
    use raads_engines::pde1d::{self, Grid, PdeModel};
    const { assert!(SmoothObstacle::OBSTACLE) };
    assert_eq!(SmoothObstacle::OBSTACLE_EPS, 0.01);
    const { assert!(<SmoothObstacle as PdeModel>::OBSTACLE) };
    assert_eq!(<SmoothObstacle as PdeModel>::OBSTACLE_EPS, 0.01);
    const { assert!(!<Heat<Half> as PdeModel>::OBSTACLE) };
    assert_eq!(<Heat<Half> as PdeModel>::OBSTACLE_EPS, 0.0);

    // the obstacle holds the solution near its peak: u(0, T) ≈ 1
    let v = pde1d::value::<SmoothObstacle, f64>(&[0.5, 0.0, 1.0], &Grid::new(-4.0, 4.0, 201, 200));
    assert!((v - 1.0).abs() < 0.02, "{v}");
}

#[test]
fn generic_models() {
    use raads_engines::{paths, pde1d};
    // u(x0, T) = exp(-x0² / (1 + 4kT)) / sqrt(1 + 4kT)
    let v = pde1d::value::<Heat<Half>, f64>(&[0.5, 1.0], &pde1d::Grid::new(-6.0, 6.0, 401, 400));
    let exact = (-0.25f64 / 3.0).exp() / 3f64.sqrt();
    assert!((v - exact).abs() < 1e-4, "{v} vs {exact}");

    let p = [100.5, 2.0, 1.0];
    assert_eq!(Hinge::<100>::functional([99.0], p, [0.0]), 0.0);
    let n_paths = 4096;
    let normals: Vec<f32> = (0..n_paths).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
    let r = paths::gradient::<Hinge<100>, f64>(&p, 1, n_paths, &normals);
    let s = Hinge::<100>::init(p);
    let want = Hinge::<100>::functional(Hinge::<100>::step(s, Hinge::<100>::setup(p, 1.0), [0.5]), p, [0.0]) / 2.0;
    assert!((r.value - want).abs() < 1e-12, "{} vs {want}", r.value);
}
