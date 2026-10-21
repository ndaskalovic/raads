//! Enzyme (`std::autodiff`, reverse mode at the LLVM level) on the comparison workloads, f64,
//! single core. Two versions of each first-order workload: plain Rust with one attribute (the
//! undifferentiated functions are the plain-Rust value-only reference), and hand-vectorised with
//! `std::simd` (16 lanes, explicit FMA, a SIMD `exp`).
#![feature(autodiff, portable_simd)]

use raads_bench::*;
use std::autodiff::{autodiff_forward, autodiff_reverse};
use std::hint::black_box;
use std::simd::StdFloat;
use std::simd::prelude::*;

type S = Simd<f64, MC_W>;

/// 1/k!, k = 0..=12.
const INV_FACT: [f64; 13] = {
    let mut p = [1.0; 13];
    let mut k = 1;
    while k < 13 {
        p[k] = p[k - 1] / k as f64;
        k += 1;
    }
    p
};

/// SIMD `exp` with raads' f64 algorithm: `2^n · p(r)`, two-constant Cody-Waite reduction,
/// degree-12 Taylor polynomial in FMA Horner form, `x` clamped to [-700, 700].
#[inline(always)]
fn exp_simd(x: S) -> S {
    let x = x.simd_clamp(S::splat(-700.0), S::splat(700.0));
    let n = (x * S::splat(std::f64::consts::LOG2_E)).round();
    let r = n.mul_add(S::splat(-0.693_145_751_953_125), x);
    let r = n.mul_add(S::splat(-1.428_606_820_309_417_3e-6), r);
    let mut p = S::splat(INV_FACT[12]);
    for k in (0..12).rev() {
        p = p.mul_add(r, S::splat(INV_FACT[k]));
    }
    let scale = S::from_bits(((n.cast::<i64>() + Simd::splat(1023)) << Simd::splat(52)).cast::<u64>());
    p * scale
}

/// Mean discounted payoff of a GBM call option; paths in blocks of `MC_W` lanes.
#[inline(always)]
fn mean_payoff(p: &[f64; 5], z: &[f32], payoff: impl Fn(f64) -> f64) -> f64 {
    let (s0, r, sigma, t, k) = (p[0], p[1], p[2], p[3], p[4]);
    let dt = t / MC_STEPS as f64;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let x0 = s0.ln();
    let mut acc = [0.0; MC_W];
    for block in z.chunks_exact(MC_STEPS * MC_W) {
        let mut x = [x0; MC_W];
        for zk in block.chunks_exact(MC_W) {
            for l in 0..MC_W {
                x[l] = x[l] + drift + vol * zk[l] as f64;
            }
        }
        for l in 0..MC_W {
            acc[l] += payoff(x[l].exp() - k);
        }
    }
    acc.iter().sum::<f64>() / MC_PATHS as f64 * (-r * t).exp()
}

#[autodiff_reverse(d_price, Duplicated, Const, Active)]
fn price(p: &[f64; 5], z: &[f32]) -> f64 {
    mean_payoff(p, z, |v| v.max(0.0))
}

/// `price` hand-vectorised: one `Simd` of 16 paths per block (`exp_simd` beat `std::simd`'s `exp` here).
#[autodiff_reverse(d_price_simd, Duplicated, Const, Active)]
fn price_simd(p: &[f64; 5], z: &[f32]) -> f64 {
    let (s0, r, sigma, t, k) = (p[0], p[1], p[2], p[3], p[4]);
    let dt = t / MC_STEPS as f64;
    let drift = S::splat((r - 0.5 * sigma * sigma) * dt);
    let vol = S::splat(sigma * dt.sqrt());
    let x0 = S::splat(s0.ln());
    let mut acc = S::splat(0.0);
    for block in z.chunks_exact(MC_STEPS * MC_W) {
        let mut x = x0;
        for zk in block.chunks_exact(MC_W) {
            x = vol.mul_add(Simd::<f32, MC_W>::from_slice(zk).cast(), x + drift);
        }
        acc += (exp_simd(x) - S::splat(k)).simd_max(S::splat(0.0));
    }
    acc.reduce_sum() / MC_PATHS as f64 * (-r * t).exp()
}

#[autodiff_reverse(d_f, Active, Active, Const, Active)]
fn f(x: f64, y: f64, z: f64) -> f64 {
    (x * y).exp() + (x * x + z).sqrt() / y
}

/// `f` hand-vectorised: 16 points per call. `std::simd`'s `exp` here: `exp_simd` made this 2× slower.
#[autodiff_reverse(d_f_simd, Duplicated, Duplicated, Const, Duplicated)]
fn f_simd(x: &S, y: &S, z: &S, out: &mut S) {
    *out = (x * y).exp() + x.mul_add(*x, *z).sqrt() / y;
}

/// Gradient and H·v at one point: forward mode over the reverse-mode `d_f`.
#[autodiff_forward(d_grad_f, Dual, Dual, Const, Dual)]
fn grad_f1(x: f64, y: f64, z: f64, out: &mut [f64; 2]) {
    let (_, dx, dy) = d_f(x, y, z, 1.0);
    *out = [dx, dy];
}

/// SABR lognormal implied volatility (Hagan et al. 2002). p = [F, α, ρ, ν].
#[autodiff_reverse(d_sabr, Duplicated, Const, Const, Const, Active)]
fn sabr(p: &[f64; 4], k: f64, t: f64, beta: f64) -> f64 {
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

/// `out` = [vol, gradient]; its forward-mode tangent along `p`'s tangent is [∇·v, H·v].
#[autodiff_forward(d_grad_sabr, Dual, Const, Const, Const, Dual)]
fn grad_sabr(p: &[f64; 4], k: f64, t: f64, beta: f64, out: &mut [f64; 5]) {
    let mut g = [0.0; 4];
    out[0] = d_sabr(p, &mut g, k, t, beta, 1.0);
    out[1..].copy_from_slice(&g);
}

fn hv_sabr(s: &SabrPoints, h: &mut [Vec<f64>; 9]) {
    for i in 0..s.k.len() {
        let p = [s.p[0][i], s.p[1][i], s.p[2][i], s.p[3][i]];
        let (mut out, mut t) = ([0.0; 5], [0.0; 5]);
        d_grad_sabr(&p, &SABR_V, s.k[i], s.t[i], SABR_BETA, &mut out, &mut t);
        h[0][i] = out[0];
        for j in 0..4 {
            h[1 + j][i] = out[1 + j];
            h[5 + j][i] = t[1 + j];
        }
    }
}

fn mc(z: &[f32]) -> (f64, [f64; 5]) {
    let mut g = [0.0; 5];
    let v = d_price(black_box(&MC_PARAMS), &mut g, z, 1.0);
    (v, g)
}

fn mc_simd(z: &[f32]) -> (f64, [f64; 5]) {
    let mut g = [0.0; 5];
    let v = d_price_simd(black_box(&MC_PARAMS), &mut g, z, 1.0);
    (v, g)
}

fn grad_f_simd(x: &[f64], y: &[f64], z: &[f64], v: &mut [f64], gx: &mut [f64], gy: &mut [f64]) {
    for i in (0..x.len()).step_by(MC_W) {
        let (xs, ys, zs) = (S::from_slice(&x[i..]), S::from_slice(&y[i..]), S::from_slice(&z[i..]));
        let (mut dx, mut dy, mut o, mut seed) = (S::splat(0.0), S::splat(0.0), S::splat(0.0), S::splat(1.0));
        d_f_simd(&xs, &mut dx, &ys, &mut dy, &zs, &mut o, &mut seed);
        o.copy_to_slice(&mut v[i..i + MC_W]);
        dx.copy_to_slice(&mut gx[i..i + MC_W]);
        dy.copy_to_slice(&mut gy[i..i + MC_W]);
    }
}

fn hv_f(x: &[f64], y: &[f64], z: &[f64], g: &mut [Vec<f64>; 4]) {
    for i in 0..x.len() {
        let (mut out, mut t) = ([0.0; 2], [0.0; 2]);
        d_grad_f(x[i], F_V[0], y[i], F_V[1], z[i], &mut out, &mut t);
        [g[0][i], g[1][i], g[2][i], g[3][i]] = [out[0], out[1], t[0], t[1]];
    }
}

fn grad_f(x: &[f64], y: &[f64], z: &[f64], v: &mut [f64], gx: &mut [f64], gy: &mut [f64]) {
    for i in 0..x.len() {
        (v[i], gx[i], gy[i]) = d_f(x[i], y[i], z[i], 1.0);
    }
}

fn main() {
    let z = normals();
    let (x, y, zd) = points();
    let n = F_POINTS;
    let (mut v, mut gx, mut gy) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut h: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; n]);
    let sp = sabr_points();
    let mut hs: [Vec<f64>; 9] = std::array::from_fn(|_| vec![0.0; n]);

    let (p, g) = mc(&z);
    report_mc("enzyme", p, &g);
    let (p, g) = mc_simd(&z);
    report_mc("enz-simd", p, &g);
    grad_f(&x, &y, &zd, &mut v, &mut gx, &mut gy);
    report_f("enzyme", &v, &gx, &gy);
    grad_f_simd(&x, &y, &zd, &mut v, &mut gx, &mut gy);
    report_f("enz-simd", &v, &gx, &gy);
    hv_f(&x, &y, &zd, &mut h);
    report_f_hv("enzyme", &h);
    hv_sabr(&sp, &mut hs);
    report_sabr("enzyme", &hs);

    let (ms, _) = best_ms(|| {
        let (p, g) = mc(&z);
        p + g.iter().sum::<f64>()
    });
    report_time("enzyme", "mc price + 5 Greeks", ms);
    let (ms, _) = best_ms(|| price(black_box(&MC_PARAMS), &z));
    report_time("plain", "mc price only", ms);
    let (ms, _) = best_ms(|| {
        let (p, g) = mc_simd(&z);
        p + g.iter().sum::<f64>()
    });
    report_time("enz-simd", "mc price + 5 Greeks", ms);
    let (ms, _) = best_ms(|| price_simd(black_box(&MC_PARAMS), &z));
    report_time("simd", "mc price only", ms);
    let (ms, _) = best_ms(|| {
        grad_f(black_box(&x), &y, &zd, &mut v, &mut gx, &mut gy);
        v[n - 1] + gx[n - 1] + gy[n - 1]
    });
    report_time("enzyme", "f value + gradient", ms);
    let (ms, _) = best_ms(|| {
        grad_f_simd(black_box(&x), &y, &zd, &mut v, &mut gx, &mut gy);
        v[n - 1] + gx[n - 1] + gy[n - 1]
    });
    report_time("enz-simd", "f value + gradient", ms);
    let (ms, _) = best_ms(|| {
        let x = black_box(&x);
        for i in 0..n {
            v[i] = f(x[i], y[i], zd[i]);
        }
        v[n - 1]
    });
    report_time("plain", "f value only", ms);
    let (ms, _) = best_ms(|| {
        hv_f(black_box(&x), &y, &zd, &mut h);
        h.iter().map(|a| a[n - 1]).sum()
    });
    report_time("enzyme", "f gradient + H·v", ms);
    let (ms, _) = best_ms(|| {
        hv_sabr(black_box(&sp), &mut hs);
        hs.iter().map(|a| a[n - 1]).sum()
    });
    report_time("enzyme", "sabr vol + gradient + H·v", ms);
}
