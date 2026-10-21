//! raads on the comparison workloads (see `lib.rs`), f64, single core.

use raads::lanes::{F64x8, F64x16};
use raads::prelude::*;
use raads_bench::*;
use raads_engines::paths;
use raads_finance::gbm::GbmCall;
use std::hint::black_box;

#[kernel]
pub fn f<N: AdNum>(v: [N; 2], #[data] z: N) -> N {
    let (x, y) = (v[0], v[1]);
    (x * y).exp() + (x * x + z).sqrt() / y
}

/// SABR lognormal implied volatility (Hagan et al. 2002). p = [F, α, ρ, ν].
#[kernel]
pub fn sabr<N: AdNum>(p: [N; 4], #[data] k: N, #[data] t: N, #[data] beta: N) -> N {
    let (f, alpha, rho, nu) = (p[0], p[1], p[2], p[3]);
    let omb = 1.0 - beta;
    let lfk = (f / k).ln();
    let fkb = (0.5 * omb * (f * k).ln()).exp(); // (F K)^((1 - β) / 2)
    let z = nu / alpha * fkb * lfk;
    let x = (((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho) / (1.0 - rho)).ln();
    let omb2 = omb * omb;
    let l2 = lfk * lfk;
    let denom = fkb * (1.0 + omb2 / 24.0 * l2 + omb2 * omb2 / 1920.0 * l2 * l2);
    let corr = 1.0
        + t * (omb2 * alpha * alpha / (24.0 * fkb * fkb)
            + rho * beta * nu * alpha / (4.0 * fkb)
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0);
    alpha / denom * (z / x) * corr
}

type L = F64x16;
/// `Dual` lanes for H·v (8 measured faster than 16 here).
type D = Dual<F64x8>;

fn mc(z: &[f32]) -> (f64, [f64; 5]) {
    let p = MC_PARAMS.map(L::from);
    let r = paths::gradient::<GbmCall, L>(black_box(&p), MC_STEPS, MC_PATHS, z);
    (r.value, r.grad)
}

/// Value and gradient at every point, `W` points per call.
fn grad_f(x: &[f64], y: &[f64], z: &[f64], v: &mut [f64], gx: &mut [f64], gy: &mut [f64]) {
    for i in (0..x.len()).step_by(L::W) {
        let mut g = [L::from(0.0); 2];
        let val = f::vjp(&[L::load(&x[i..]), L::load(&y[i..])], L::load(&z[i..]), L::from(1.0), &mut g);
        val.store(&mut v[i..]);
        g[0].store(&mut gx[i..]);
        g[1].store(&mut gy[i..]);
    }
}

fn dual(v: F64x8, t: f64) -> D {
    D::new(v, F64x8::from(t))
}

/// Gradient and H·v at every point (forward-over-reverse on `Dual` lanes).
fn hv_f(x: &[f64], y: &[f64], z: &[f64], g: &mut [Vec<f64>; 4]) {
    for i in (0..x.len()).step_by(F64x8::W) {
        let p = [dual(F64x8::load(&x[i..]), F_V[0]), dual(F64x8::load(&y[i..]), F_V[1])];
        let mut a = [D::from(0.0); 2];
        f::vjp(&p, dual(F64x8::load(&z[i..]), 0.0), D::from(1.0), &mut a);
        a[0].v.store(&mut g[0][i..]);
        a[1].v.store(&mut g[1][i..]);
        a[0].t.store(&mut g[2][i..]);
        a[1].t.store(&mut g[3][i..]);
    }
}

/// SABR vol, gradient and H·v at every point: `h` = [vol, grad (4), H·v (4)].
fn hv_sabr(s: &SabrPoints, h: &mut [Vec<f64>; 9]) {
    for i in (0..s.k.len()).step_by(F64x8::W) {
        let p: [D; 4] = std::array::from_fn(|j| dual(F64x8::load(&s.p[j][i..]), SABR_V[j]));
        let (k, t) = (dual(F64x8::load(&s.k[i..]), 0.0), dual(F64x8::load(&s.t[i..]), 0.0));
        let mut a = [D::from(0.0); 4];
        let vol = sabr::vjp(&p, k, t, D::from(SABR_BETA), D::from(1.0), &mut a);
        vol.v.store(&mut h[0][i..]);
        for j in 0..4 {
            a[j].v.store(&mut h[1 + j][i..]);
            a[j].t.store(&mut h[5 + j][i..]);
        }
    }
}

fn main() {
    assert_eq!(L::W, MC_W);
    let z = normals();
    let (x, y, zd) = points();
    let sp = sabr_points();
    let n = F_POINTS;
    let (mut v, mut gx, mut gy) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut h: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; n]);
    let mut hs: [Vec<f64>; 9] = std::array::from_fn(|_| vec![0.0; n]);

    let (price, g) = mc(&z);
    report_mc("raads", price, &g);
    grad_f(&x, &y, &zd, &mut v, &mut gx, &mut gy);
    report_f("raads", &v, &gx, &gy);
    hv_f(&x, &y, &zd, &mut h);
    report_f_hv("raads", &h);
    hv_sabr(&sp, &mut hs);
    report_sabr("raads", &hs);

    let (ms, _) = best_ms(|| {
        let (p, g) = mc(&z);
        p + g.iter().sum::<f64>()
    });
    report_time("raads", "mc price + 5 Greeks", ms);
    let (ms, _) = best_ms(|| {
        let p = MC_PARAMS.map(L::from);
        paths::value::<GbmCall, L>(black_box(&p), MC_STEPS, MC_PATHS, &z).value
    });
    report_time("raads", "mc price only", ms);
    let (ms, _) = best_ms(|| {
        grad_f(black_box(&x), &y, &zd, &mut v, &mut gx, &mut gy);
        v[n - 1] + gx[n - 1] + gy[n - 1]
    });
    report_time("raads", "f value + gradient", ms);
    let (ms, _) = best_ms(|| {
        for i in (0..n).step_by(L::W) {
            f::eval(&[L::load(&black_box(&x)[i..]), L::load(&y[i..])], L::load(&zd[i..])).store(&mut v[i..]);
        }
        v[n - 1]
    });
    report_time("raads", "f value only", ms);
    let (ms, _) = best_ms(|| {
        hv_f(black_box(&x), &y, &zd, &mut h);
        h.iter().map(|a| a[n - 1]).sum()
    });
    report_time("raads", "f gradient + H·v", ms);
    let (ms, _) = best_ms(|| {
        hv_sabr(black_box(&sp), &mut hs);
        hs.iter().map(|a| a[n - 1]).sum()
    });
    report_time("raads", "sabr vol + gradient + H·v", ms);
}
