//! FD engine, f64: forward == reverse,
//! gradient == finite differences, European call vs Black-Scholes, American put vs a CRR
//! binomial tree, lanes == single solves.
//!
//! Also: lanes with different spots (one clamped to the grid edge), with hard and smoothed
//! exercise; f32 vs f64 Greeks; the argument checks.

use raads::lanes::{F32x4, F64x4};
use raads::{Arr, Dual, Lanes};
use raads_engines::pde1d::{self, Grid, PdeModel};
use raads_finance::bs_fd::{BsAmericanPut, BsAmericanPutSmooth, BsEuroCall};

mod common;
use common::{Checks, rel};

type D = Dual<f64>;

fn crr_american_put(s: f64, k: f64, r: f64, sigma: f64, t: f64, n: usize) -> f64 {
    let dt = t / n as f64;
    let up = (sigma * dt.sqrt()).exp();
    let dn = 1.0 / up;
    let q = ((r * dt).exp() - dn) / (up - dn);
    let disc = (-r * dt).exp();
    let node = |i: usize, step: usize| s * up.powi(i as i32) * dn.powi((step - i) as i32);
    let mut v: Vec<f64> = (0..=n).map(|i| (k - node(i, n)).max(0.0)).collect();
    for step in (0..n).rev() {
        for i in 0..=step {
            v[i] = (disc * (q * v[i + 1] + (1.0 - q) * v[i])).max(k - node(i, step));
        }
    }
    v[0]
}

fn params<M: PdeModel>(p: &[f64]) -> M::P<f64> {
    Arr::from_fn(|i| p[i])
}

fn verify<M: PdeModel>(c: &mut Checks, label: &str, p: &[f64], g: &Grid, fd_tol: f64) {
    println!("{label} {}x{}", g.nx, g.nt);
    let r = pde1d::gradient::<M, f64>(&params::<M>(p), g);
    // dot test: forward tangent of the price through the scheme vs the reverse sweep
    let v = |j: usize| 0.3 + 0.1 * j as f64;
    let pd: M::P<D> = Arr::from_fn(|j| D::new(p[j], v(j)));
    let gv: f64 = r.grad.as_ref().iter().enumerate().map(|(j, g)| g * v(j)).sum();
    let d = pde1d::gradient::<M, D>(&pd, g);
    c.le(&format!("{label}: dot test (rel)"), (d.value.t - gv).abs() / gv.abs(), 1e-12);
    // gradient vs central differences of the price-only solve
    let mut e = 0.0f64;
    for j in 0..p.len() {
        let h = 1e-6 * p[j].abs().max(1e-2);
        let (mut pp, mut pm) = (p.to_vec(), p.to_vec());
        pp[j] += h;
        pm[j] -= h;
        let fd = (pde1d::value::<M, f64>(&params::<M>(&pp), g) - pde1d::value::<M, f64>(&params::<M>(&pm), g)) / (2.0 * h);
        e = e.max((r.grad.as_ref()[j] - fd).abs() / fd.abs().max(1e-3));
    }
    c.le(&format!("{label}: gradient vs central differences (rel)"), e, fd_tol);
}

#[test]
fn fd_engine() {
    let mut c = Checks::default();
    let p = [100.0, 0.01, 0.2, 1.0, 100.0];
    let (lo, hi) = (100f64.ln() - 6.0 * 0.2, 100f64.ln() + 6.0 * 0.2);
    let g201 = Grid::new(lo, hi, 201, 200);
    verify::<BsEuroCall>(&mut c, "European call", &p, &g201, 1e-6);
    verify::<BsAmericanPut>(&mut c, "American put", &p, &g201, 1e-6);

    let g801 = Grid::new(lo, hi, 801, 800);
    let e = pde1d::gradient::<BsEuroCall, f64>(&p, &g801);
    let bs = [8.433318690109608, 0.5596176923702425, 47.52845054691464, 39.447933090788894, 4.420077814548036, -0.4752845054691464];
    let mut worst = (e.value - bs[0]).abs() / bs[0];
    for j in 0..5 {
        worst = worst.max((e.grad[j] - bs[j + 1]).abs() / bs[j + 1].abs());
    }
    println!("European call 801x800 vs Black-Scholes");
    c.le("price and 5 Greeks (rel)", worst, 2e-5);
    let a = pde1d::value::<BsAmericanPut, f64>(&p, &g801);
    let crr = crr_american_put(100.0, 100.0, 0.01, 0.2, 1.0, 4000);
    println!("American put 801x800 vs CRR(4000): {a} {crr}");
    c.le("price (rel)", (a - crr).abs() / crr, 5e-5);

    // 4 strikes in 4 lanes == 4 single solves; pde1d::value == pde1d::gradient value
    let strikes = [90.0, 100.0, 110.0, 120.0];
    let mut pv = p.map(F64x4::from);
    pv[4] = F64x4::from_fn(|l| strikes[l]);
    let bt = pde1d::gradient::<BsAmericanPut, F64x4>(&pv, &g201);
    let po = pde1d::value::<BsAmericanPut, F64x4>(&pv, &g201);
    let mut lane_err = 0.0f64;
    for (k, &kk) in strikes.iter().enumerate() {
        let mut q = p;
        q[4] = kk;
        let sgl = pde1d::gradient::<BsAmericanPut, f64>(&q, &g201);
        lane_err = lane_err.max((bt.value.lane(k) - sgl.value).abs() / sgl.value).max((po.lane(k) - sgl.value).abs() / sgl.value);
        for j in 0..5 {
            lane_err = lane_err.max((bt.grad[j].lane(k) - sgl.grad[j]).abs() / (sgl.grad[j].abs() + 1e-12));
        }
    }
    println!("4 strikes in lanes vs single solves");
    c.le("price and gradient (rel)", lane_err, 1e-12);
    c.assert_ok();
}

/// Lanes with different spots (different interpolation nodes; 400 lies beyond `x_hi`, so its
/// node is clamped) == single solves, and the W = 4 Dual dot test per lane. Gradients are
/// compared relative to max(|g|, 1e-3): the smoothed exercise calls `exp`, which differs in the
/// last bit between the lanes (polynomial) and `f64` (libm), and some Greeks are ~1e-8 here.
fn lanes_vs_single<M: PdeModel>(c: &mut Checks, label: &str) {
    let g = Grid::new(100f64.ln() - 1.2, 100f64.ln() + 1.2, 201, 200);
    let spots = [55.0, 100.0, 131.7, 400.0];
    let base = [100.0, 0.01, 0.2, 1.0, 100.0];
    let pv: M::P<F64x4> = Arr::from_fn(|j| if j == 0 { F64x4::from_fn(|l| spots[l]) } else { F64x4::from(base[j]) });
    let r = pde1d::gradient::<M, F64x4>(&pv, &g);
    let v = |j: usize| 0.3 + 0.1 * j as f64;
    let pd: M::P<Dual<F64x4>> = Arr::from_fn(|j| Dual::new(pv.as_ref()[j], F64x4::from(v(j))));
    let d = pde1d::gradient::<M, Dual<F64x4>>(&pd, &g);
    let (mut e, mut ed) = (0.0f64, 0.0f64);
    for (l, &s) in spots.iter().enumerate() {
        let sg = pde1d::gradient::<M, f64>(&Arr::from_fn(|j| if j == 0 { s } else { base[j] }), &g);
        e = e.max(rel(r.value.lane(l), sg.value));
        for j in 0..5 {
            e = e.max((r.grad.as_ref()[j].lane(l) - sg.grad.as_ref()[j]).abs() / sg.grad.as_ref()[j].abs().max(1e-3));
        }
        let gv: f64 = (0..5).map(|j| r.grad.as_ref()[j].lane(l) * v(j)).sum();
        ed = ed.max(rel(d.value.t.lane(l), gv));
    }
    println!("{label}");
    c.le("lanes (different spots) vs single solves (rel)", e, 1e-11);
    c.le("W=4 Dual dot test (rel)", ed, 1e-10);
}

#[test]
fn fd_lanes_different_spots() {
    let mut c = Checks::default();
    lanes_vs_single::<BsEuroCall>(&mut c, "European call");
    lanes_vs_single::<BsAmericanPut>(&mut c, "American put");
    lanes_vs_single::<BsAmericanPutSmooth>(&mut c, "American put, smoothed exercise");
    c.assert_ok();
}

/// f32 lanes on 201x200: price and Greeks within 1e-4 of f64 (f32 is for coarse grids).
#[test]
fn fd_f32_vs_f64() {
    let g = Grid::new(100f64.ln() - 1.2, 100f64.ln() + 1.2, 201, 200);
    let p = [100.0, 0.01, 0.2, 1.0, 100.0];
    let mut c = Checks::default();
    for american in [false, true] {
        let (r64, r32) = if american {
            (pde1d::gradient::<BsAmericanPut, f64>(&p, &g), pde1d::gradient::<BsAmericanPut, F32x4>(&p.map(F32x4::from), &g))
        } else {
            (pde1d::gradient::<BsEuroCall, f64>(&p, &g), pde1d::gradient::<BsEuroCall, F32x4>(&p.map(F32x4::from), &g))
        };
        let label = if american { "American put" } else { "European call" };
        c.rel(&format!("{label} price"), r32.value.lane(0), r64.value, 1e-4);
        for j in 0..5 {
            c.rel(&format!("{label} grad {j}"), r32.grad[j].lane(0), r64.grad[j], 1e-4);
        }
    }
    c.assert_ok();
}

#[test]
#[should_panic(expected = "need at least 1 time step")]
fn fd_zero_steps() {
    let _ = pde1d::gradient::<BsEuroCall, f64>(&[100.0, 0.01, 0.2, 1.0, 100.0], &Grid::new(3.0, 6.0, 51, 0));
}

#[test]
#[should_panic(expected = "need at least 3 grid nodes")]
fn fd_two_nodes() {
    let _ = pde1d::value::<BsEuroCall, f64>(&[100.0, 0.01, 0.2, 1.0, 100.0], &Grid::new(3.0, 6.0, 2, 10));
}

#[test]
#[should_panic(expected = "need finite x_lo < x_hi")]
fn fd_empty_range() {
    let _ = pde1d::value::<BsEuroCall, f64>(&[100.0, 0.01, 0.2, 1.0, 100.0], &Grid::new(6.0, 3.0, 51, 10));
}

#[test]
#[should_panic(expected = "need theta in [0, 1]")]
fn fd_theta_out_of_range() {
    let _ = pde1d::value::<BsEuroCall, f64>(&[100.0, 0.01, 0.2, 1.0, 100.0], &Grid::new(3.0, 6.0, 51, 10).with_theta(1.5));
}

/// Fully implicit vs Crank–Nicolson: first order in dt, so close but not equal.
#[test]
fn fd_with_theta() {
    let p = [100.0, 0.01, 0.2, 1.0, 100.0];
    let g = Grid::new(100f64.ln() - 1.5, 100f64.ln() + 1.5, 201, 400);
    let (cn, imp) = (pde1d::value::<BsEuroCall, f64>(&p, &g), pde1d::value::<BsEuroCall, f64>(&p, &g.with_theta(1.0)));
    assert!(cn != imp && (cn - imp).abs() < 1e-2, "{cn} vs {imp}");
}
