//! 1-D PDE engine, f64:
//! - heat equation with drift and decay vs its closed form (value, gradient, and with
//!   `snapshots` every read at several horizons);
//! - dot test (Dual forward tangent == reverse gradient · v) and gradient == central
//!   differences of `value`, with and without the obstacle;
//! - obstacle problem vs an independent explicit scheme with projection on a finer grid;
//! - Hessian in one pass (`DualVec`, unit seeds) == one `Dual` pass per column, and symmetric;
//! - `snapshots_weighted` gradient == forward Dual tangents (W = 2 lanes, every Rannacher
//!   count, one or several horizons); the argument checks.

use raads::lanes::F64x2;
use raads::{AdNum, Arr, Dual, DualVec, Lanes, kernels};
use raads_engines::pde1d::{self, Grid, PdeModel, Read, SnapshotGrid};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// u_τ = k u_xx + b u_x - c u, u(x, 0) = exp(-x²). params: k, b, c, x0, T.
pub struct Heat;

#[kernels(raads_engines::pde1d)]
impl Heat {
    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
        [p[0], p[1], p[2]]
    }
    pub fn initial<N: AdNum>(#[data] x: N, _p: [N; 5]) -> N {
        (-(x * x)).exp()
    }
    pub fn x0<N: AdNum>(p: [N; 5]) -> N {
        p[3]
    }
    pub fn horizon<N: AdNum>(p: [N; 5]) -> N {
        p[4]
    }
}

/// u(x, τ) = exp(-cτ - (x + bτ)² / s) / √s, s = 1 + 4kτ.
fn heat_exact(k: f64, b: f64, c: f64, x: f64, t: f64) -> f64 {
    let s = 1.0 + 4.0 * k * t;
    (-c * t - (x + b * t).powi(2) / s).exp() / s.sqrt()
}

/// u_τ = k u_xx - c u, u(x, 0) = exp(-x²), u >= exp(-x²). params: k, c, x0, T.
pub struct Decay;

#[kernels(raads_engines::pde1d)]
impl Decay {
    const OBSTACLE: bool = true;

    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 4]) -> [N; 3] {
        [p[0], N::from(0.0), p[1]]
    }
    pub fn initial<N: AdNum>(#[data] x: N, _p: [N; 4]) -> N {
        (-(x * x)).exp()
    }
    pub fn x0<N: AdNum>(p: [N; 4]) -> N {
        p[2]
    }
    pub fn horizon<N: AdNum>(p: [N; 4]) -> N {
        p[3]
    }
}

/// `Decay` by explicit Euler with projection onto the obstacle, `u_xx = 0` boundaries,
/// quadratic interpolation at x0.
fn decay_explicit(k: f64, c: f64, x0: f64, t: f64, nx: usize) -> f64 {
    let (lo, hi) = (-6.0, 6.0);
    let h = (hi - lo) / (nx - 1) as f64;
    let nt = (t / (0.4 * h * h / k)).ceil() as usize;
    let dt = t / nt as f64;
    let g: Vec<f64> = (0..nx).map(|i| (-(lo + i as f64 * h).powi(2)).exp()).collect();
    let mut u = g.clone();
    for _ in 0..nt {
        let prev = u.clone();
        for i in 0..nx {
            let uxx = if i == 0 || i == nx - 1 { 0.0 } else { (prev[i - 1] - 2.0 * prev[i] + prev[i + 1]) / (h * h) };
            u[i] = (prev[i] + dt * (k * uxx - c * prev[i])).max(g[i]);
        }
    }
    let j = ((x0 - lo) / h).round() as usize;
    let s = (x0 - (lo + j as f64 * h)) / h;
    s * (s - 1.0) * 0.5 * u[j - 1] + (1.0 - s * s) * u[j] + s * (s + 1.0) * 0.5 * u[j + 1]
}

/// Full Hessian from one `DualVec` pass == `K` `Dual` passes (unit seeds); `symmetric` also
/// checks H = Hᵀ (not for a hard-max obstacle, whose Hessian is only valid in parts).
#[allow(clippy::needless_range_loop)] // matrix indices read clearer here
fn hessian_one_pass<M: PdeModel, const K: usize>(label: &str, p: [f64; K], g: &Grid, symmetric: bool) {
    let hv = pde1d::gradient::<M, DualVec<f64, K>>(&Arr::from_fn(|i| DualVec::unit(p[i], i)), g);
    let mut h = [[0.0; K]; K];
    for k in 0..K {
        let hd = pde1d::gradient::<M, Dual<f64>>(&Arr::from_fn(|i| Dual::new(p[i], if i == k { 1.0 } else { 0.0 })), g);
        assert_eq!(hv.value.v, hd.value.v, "{label}: value");
        for i in 0..K {
            let (a, b) = (hv.grad.as_ref()[i].t[k], hd.grad.as_ref()[i].t);
            assert_eq!(hv.grad.as_ref()[i].v, hd.grad.as_ref()[i].v, "{label}: gradient {i}");
            // tangents round differently (shared reciprocals), compounded over the steps
            assert!((a - b).abs() <= 1e-10 * b.abs().max(1.0), "{label}: H[{i}][{k}] {a} vs Dual {b}");
            h[i][k] = a;
        }
    }
    for i in 0..K * symmetric as usize {
        for k in 0..i {
            assert!(
                (h[i][k] - h[k][i]).abs() <= 1e-9 * h[i][k].abs().max(1.0),
                "{label}: H[{i}][{k}] {} vs H[{k}][{i}] {}",
                h[i][k],
                h[k][i]
            );
        }
    }
}

/// Dot test and gradient vs central differences of `value`; returns the gradient.
fn verify<M: PdeModel>(label: &str, p: &[f64], g: &Grid) -> M::P<f64> {
    let r = pde1d::gradient::<M, f64>(&Arr::from_fn(|i| p[i]), g);
    let v = |j: usize| 0.3 + 0.1 * j as f64;
    let d = pde1d::gradient::<M, Dual<f64>>(&Arr::from_fn(|j| Dual::new(p[j], v(j))), g);
    let gv: f64 = r.grad.as_ref().iter().enumerate().map(|(j, g)| g * v(j)).sum();
    assert!((d.value.t - gv).abs() <= 1e-12 * gv.abs(), "{label}: dot test {} vs {gv}", d.value.t);
    assert_eq!(d.value.v, r.value);
    for j in 0..p.len() {
        let h = 1e-5 * p[j].abs().max(1e-2);
        let (mut a, mut b) = (p.to_vec(), p.to_vec());
        a[j] += h;
        b[j] -= h;
        let fd = (pde1d::value::<M, f64>(&Arr::from_fn(|i| a[i]), g) - pde1d::value::<M, f64>(&Arr::from_fn(|i| b[i]), g)) / (2.0 * h);
        let gj = r.grad.as_ref()[j];
        assert!((gj - fd).abs() <= 1e-6 * fd.abs().max(1e-3), "{label}: grad {j} {gj} vs central differences {fd}");
    }
    r.grad
}

#[test]
fn heat() {
    let p = [0.5, 0.3, 0.2, 0.4, 1.0];
    let g = Grid::new(-6.0, 6.0, 801, 800);
    let grad = verify::<Heat>("heat", &p, &g);
    hessian_one_pass::<Heat, 5>("heat", p, &g, true);
    let exact = |q: [f64; 5]| heat_exact(q[0], q[1], q[2], q[3], q[4]);
    let value = pde1d::value::<Heat, f64>(&p, &g);
    assert!((value - exact(p)).abs() < 2e-5 * exact(p), "value {value} vs {}", exact(p));
    for j in 0..5 {
        let (mut a, mut b) = (p, p);
        a[j] += 1e-6;
        b[j] -= 1e-6;
        let want = (exact(a) - exact(b)) / 2e-6;
        assert!((grad[j] - want).abs() < 2e-5, "grad {j} {} vs closed form {want}", grad[j]);
    }
}

/// The obstacle binds around x = 0 and lifts the solution at x0 = 1 above both the
/// unconstrained solution and the obstacle. Both schemes converge at first order (the contact
/// boundary); at these grids they differ by 1.2e-4, half that at twice the resolution.
#[test]
fn obstacle() {
    let p = [0.5, 0.5, 1.0, 1.0];
    let g = Grid::new(-6.0, 6.0, 801, 800);
    verify::<Decay>("obstacle", &p, &g);
    let value = pde1d::value::<Decay, f64>(&p, &g);
    let reference = decay_explicit(p[0], p[1], p[2], p[3], 1201);
    let (free, floor) = (heat_exact(p[0], 0.0, p[1], p[2], p[3]), (-p[2] * p[2]).exp());
    println!("obstacle {value} explicit reference {reference} unconstrained {free} obstacle at x0 {floor}");
    assert!(value > free + 0.1 && value > floor + 0.05, "the obstacle should lift the solution at x0");
    assert!((value - reference).abs() < 2e-4, "{value} vs explicit reference {reference}");
}

#[test]
fn snapshots_vs_closed_form() {
    let (k, b, c) = (0.5, 0.3, 0.2);
    let horizons = [0.1, 0.5, 1.0];
    let mut grid = SnapshotGrid::new(-6.0, 6.0, 1201, 1.0 / 1000.0);
    grid.rannacher = 0;
    let s = pde1d::snapshots::<Heat, f64>(&[k, b, c, 0.0, 0.0], &grid, &horizons);
    let e = 1e-4;
    let mut worst = [0.0f64; 4];
    for (m, &t) in horizons.iter().enumerate() {
        for x in [-1.0, -0.37, 0.0, 0.52, 1.5] {
            let u = |x: f64, t: f64| heat_exact(k, b, c, x, t);
            let want = [
                u(x, t),
                (u(x + e, t) - u(x - e, t)) / (2.0 * e),
                (u(x + e, t) - 2.0 * u(x, t) + u(x - e, t)) / (e * e),
                (u(x, t + e) - u(x, t - e)) / (2.0 * e),
            ];
            let [r0, r1, r2] = s.read(m, x);
            let got = [r0, r1, r2, s.read_tau(m, x)[0]];
            for i in 0..4 {
                worst[i] = worst[i].max((got[i] - want[i]).abs());
            }
        }
    }
    println!("max abs error: u {:e} u_x {:e} u_xx {:e} u_τ {:e}", worst[0], worst[1], worst[2], worst[3]);
    for (i, tol) in [1e-5, 1e-4, 2e-4, 1e-4].iter().enumerate() {
        assert!(worst[i] < *tol, "read {i}: {} > {tol}", worst[i]);
    }
}

/// `snapshots_weighted` gradient (reverse) vs forward Dual tangents, W = 2 lanes with
/// different reads and weights. Max relative error.
fn weighted_vs_tangents<M: PdeModel>(p: &[f64], rannacher: usize, horizons: &[f64]) -> f64 {
    let mut g = SnapshotGrid::new(-4.0, 4.0, 241, 1.0 / 200.0);
    g.rannacher = rannacher;
    let mut rng = StdRng::seed_from_u64(11);
    let reads: Vec<Read<F64x2>> = (0..5 * horizons.len())
        .map(|i| Read {
            horizon: i % horizons.len(),
            x: F64x2::from_fn(|_| rng.random_range(-1.0..1.0)),
            weight: F64x2::from_fn(|_| rng.random_range(-5.0..5.0)),
        })
        .collect();
    let w = pde1d::snapshots_weighted::<M, F64x2>(&Arr::from_fn(|i| F64x2::from(p[i])), &g, horizons, &reads);
    let mut e = 0.0f64;
    for j in 0..p.len() {
        let pd: M::P<Dual<F64x2>> = Arr::from_fn(|i| Dual::new(F64x2::from(p[i]), F64x2::from(if i == j { 1.0 } else { 0.0 })));
        let s = pde1d::snapshots::<M, Dual<F64x2>>(&pd, &g, horizons);
        let fwd = reads.iter().fold(F64x2::from(0.0), |f, r| f + r.weight * s.read(r.horizon, r.x)[0].t);
        for l in 0..2 {
            let (a, b) = (w.grad.as_ref()[j].lane(l), fwd.lane(l));
            e = e.max((a - b).abs() / b.abs().max(1e-9));
        }
    }
    e
}

#[test]
fn weighted_reads() {
    for rannacher in [0, 1, 2] {
        for horizons in [&[0.5][..], &[0.1, 0.3, 1.0][..]] {
            let e = weighted_vs_tangents::<Heat>(&[0.5, 0.3, 0.2, 0.0, 0.0], rannacher, horizons);
            assert!(e < 1e-11, "heat, rannacher {rannacher}, horizons {horizons:?}: {e:e}");
            let e = weighted_vs_tangents::<Decay>(&[0.5, 0.5, 0.0, 0.0], rannacher, horizons);
            assert!(e < 1e-11, "obstacle, rannacher {rannacher}, horizons {horizons:?}: {e:e}");
        }
    }
}

const P: [f64; 5] = [0.5, 0.3, 0.2, 0.4, 1.0];

#[test]
#[should_panic(expected = "need at least 1 time step")]
fn zero_steps() {
    let _ = pde1d::gradient::<Heat, f64>(&P, &Grid::new(-3.0, 3.0, 51, 0));
}

#[test]
#[should_panic(expected = "need at least 3 grid nodes")]
fn two_nodes() {
    let _ = pde1d::value::<Heat, f64>(&P, &Grid::new(-3.0, 3.0, 2, 10));
}

#[test]
#[should_panic(expected = "need finite x_lo < x_hi")]
fn empty_range() {
    let _ = pde1d::value::<Heat, f64>(&P, &Grid::new(3.0, -3.0, 51, 10));
}

#[test]
#[should_panic(expected = "need theta in [0, 1]")]
fn theta_out_of_range() {
    let _ = pde1d::value::<Heat, f64>(&P, &Grid::new(-3.0, 3.0, 51, 10).with_theta(-0.5));
}

#[test]
#[should_panic(expected = "need at least 5 grid nodes")]
fn snapshots_four_nodes() {
    let _ = pde1d::snapshots::<Heat, f64>(&P, &SnapshotGrid::new(-1.0, 1.0, 4, 0.1), &[0.5]);
}

#[test]
#[should_panic(expected = "need at least 1 horizon")]
fn snapshots_no_horizons() {
    let _ = pde1d::snapshots::<Heat, f64>(&P, &SnapshotGrid::new(-1.0, 1.0, 51, 0.1), &[]);
}

#[test]
#[should_panic(expected = "need a finite time step dt > 0")]
fn snapshots_zero_dt() {
    let _ = pde1d::snapshots::<Heat, f64>(&P, &SnapshotGrid::new(-1.0, 1.0, 51, 0.0), &[0.5]);
}

#[test]
#[should_panic(expected = "Read::horizon out of range")]
fn read_out_of_range() {
    let reads = [Read { horizon: 1, x: 0.0, weight: 1.0 }];
    let _ = pde1d::snapshots_weighted::<Heat, f64>(&P, &SnapshotGrid::new(-1.0, 1.0, 51, 0.1), &[0.5], &reads);
}

#[test]
#[should_panic(expected = "horizons must be positive and strictly increasing")]
fn horizons_decreasing() {
    let _ = pde1d::snapshots::<Heat, f64>(&P, &SnapshotGrid::new(-1.0, 1.0, 51, 0.1), &[0.5, 0.2]);
}
