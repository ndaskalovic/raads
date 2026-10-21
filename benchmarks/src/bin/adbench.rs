//! raads on the Heston MC, SABR calibration and LIBOR swaption workloads of auto-differentiation/ad-benchmarks
//! (the suite in XAD's README), with that suite's inputs, sizes and timing: median of 10 runs
//! after 3 warm-ups, f64. Inputs and XAD's results come from `ad-benchmarks/reference.cpp`
//! (target/adbench.bin, target/adbench_reference.txt); results are checked against XAD first.
//! Prints every lane width and writes the suite's CSV rows to target/adbench_raads.csv: `raads`
//! (1 lane) and `raads SIMD` (8 lanes for Heston, 4 for SABR and LIBOR). Run by
//! `ad-benchmarks/run.sh`.

use raads::lanes::{F64x2, F64x4, F64x8, F64x16};
use raads::prelude::*;
use raads_engines::paths;
use std::hint::black_box;
use std::time::Instant;

const PATHS: usize = 10_000;
const STEPS: usize = 100;
const EXPIRIES: usize = 5;
const STRIKES: usize = 20;
const ITERS: usize = 500;
/// LIBOR rates (src/libor_swaption.hpp: 80) and time steps (normals per path).
const RATES: usize = 80;
const LSTEPS: usize = 40;
/// S0, K, T, r, v0, kappa, theta, xi (src/heston.hpp)
const HESTON: [f64; 8] = [100.0, 105.0, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3];

/// Heston, Euler with reflection (`|v|`), European call (src/heston.hpp).
pub struct Heston;

#[kernels(raads_engines::paths)]
impl Heston {
    /// Correlation, fixed.
    pub const RHO: f64 = -0.7;
    /// `sqrt(1 - RHO²)` as the C++ computes it in `f64`.
    pub const RHO_C: f64 = 0.714142842854285;

    /// c: r, dt, sqrt(dt), kappa, theta, xi.
    pub fn setup<N: AdNum>(p: [N; 8], #[data] n_steps: N) -> [N; 6] {
        let dt = p[2] / n_steps;
        [p[3], dt, dt.sqrt(), p[5], p[6], p[7]]
    }

    pub fn init<N: AdNum>(p: [N; 8]) -> [N; 2] {
        [p[0], p[4]]
    }

    pub fn step<N: AdNum>(s: [N; 2], c: [N; 6], #[data] z: [N; 2]) -> [N; 2] {
        let (sp, v) = (s[0], s[1]);
        let (r, dt, sqdt, kappa, theta, xi) = (c[0], c[1], c[2], c[3], c[4], c[5]);
        let dw2 = z[0] * Self::RHO + z[1] * Self::RHO_C;
        let v_pos = N::select_gt(v, N::from(0.0), v, -v);
        let sqrt_v = (v_pos + 1e-10).sqrt();
        [sp + r * sp * dt + sqrt_v * sp * sqdt * z[0], v + kappa * (theta - v) * dt + xi * sqrt_v * sqdt * dw2]
    }

    pub fn functional<N: AdNum>(s: [N; 2], p: [N; 8], _c: [N; 6]) -> N {
        (-p[3] * p[2]).exp() * (s[0] - p[1]).relu()
    }
}

/// One LIBOR market model path and the swaption portfolio's value on it
/// (src/libor_swaption.hpp, after Giles). p: `delta, lambda[80], L0[80]`.
#[kernel]
pub fn libor<N: AdNum>(p: [N; 161], #[data] z: [N; 40]) -> N {
    let delta = p[0];
    let mut l = [N::from(0.0); 80];
    for i in 0..80 {
        l[i] = p[81 + i];
    }
    for n in 0..40 {
        let sqez = delta.sqrt() * z[n];
        let mut v = N::from(0.0);
        for i in n + 1..80 {
            let lam = p[1 + i - n - 1];
            let con1 = delta * lam;
            v = v + con1 * l[i] / (delta * l[i] + 1.0);
            l[i] = l[i] * (con1 * v + lam * (sqez - con1 * 0.5)).exp();
        }
    }
    // discount factors and annuities from the end of the path
    let mut b = N::from(1.0);
    let mut s = N::from(0.0);
    let mut bs = [N::from(0.0); 80];
    let mut ss = [N::from(0.0); 80];
    for n in 40..80 {
        b = b / (delta * l[n] + 1.0);
        s = s + delta * b;
        bs[n] = b;
        ss[n] = s;
    }
    // receiver swaptions: maturities 4, 8, 20, 28, 40 (index 39 + maturity), swap rates
    // 4.5%, 5%, 5.5%
    let mut v = N::from(0.0);
    v = v - (bs[43] + ss[43] * 0.045 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[43] + ss[43] * 0.05 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[43] + ss[43] * 0.055 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[47] + ss[47] * 0.045 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[47] + ss[47] * 0.05 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[47] + ss[47] * 0.055 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[59] + ss[59] * 0.045 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[59] + ss[59] * 0.05 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[59] + ss[59] * 0.055 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[67] + ss[67] * 0.045 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[67] + ss[67] * 0.05 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[67] + ss[67] * 0.055 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[79] + ss[79] * 0.045 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[79] + ss[79] * 0.05 - 1.0).minimum(0.0) * 100.0;
    v = v - (bs[79] + ss[79] * 0.055 - 1.0).minimum(0.0) * 100.0;
    for n in 0..40 {
        v = v / (delta * l[n] + 1.0);
    }
    v
}

/// Squared error of the Hagan 2002 vol against the market vol at one strike (src/sabr.hpp,
/// general case; the suite's strike grid has no at-the-money point). p: alpha, rho, nu.
#[kernel]
pub fn sabr_sq<N: AdNum>(p: [N; 3], #[data] f: N, #[data] k: N, #[data] t: N, #[data] beta: N, #[data] mkt: N) -> N {
    let (alpha, rho, nu) = (p[0], p[1], p[2]);
    let omb = 1.0 - beta;
    let omb2 = omb * omb;
    let omb4 = omb2 * omb2;
    let ln_fk = (f * k).ln();
    let fk_pow = (0.5 * omb * ln_fk).exp();
    let fk_pow_full = (omb * ln_fk).exp();
    let log_fk = (f / k).ln();
    let log_fk2 = log_fk * log_fk;
    let log_fk4 = log_fk2 * log_fk2;
    let alpha2 = alpha * alpha;
    let correction =
        1.0 + (omb2 / 24.0 * alpha2 / fk_pow_full + 0.25 * rho * beta * nu * alpha / fk_pow + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu) * t;
    let denom_geo = fk_pow * (1.0 + omb2 / 24.0 * log_fk2 + omb4 / 1920.0 * log_fk4);
    let z = nu / alpha * fk_pow * log_fk;
    let sqrt_term = (1.0 - 2.0 * rho * z + z * z).sqrt();
    let x_z = ((sqrt_term + z - rho) / (1.0 - rho)).ln();
    let sigma = alpha / denom_geo * (z / x_z) * correction;
    let diff = sigma - mkt;
    diff * diff
}

/// Market data per point (expiry-major), initial parameters and the perturbation schedule.
struct Sabr {
    f: Vec<f64>,
    t: Vec<f64>,
    beta: Vec<f64>,
    k: Vec<f64>,
    mkt: Vec<f64>,
    init: [f64; 15],
    dp: Vec<[f64; 15]>,
}

/// Median of `iters` timed runs after `warmup` (src/timing.hpp), in ms.
fn median_ms(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut t: Vec<f64> = (0..iters)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}

fn check(what: &str, got: &[f64], want: &[f64]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-9 * w.abs().max(1e-12), "{what}[{i}]: raads {g} vs XAD {w}");
    }
    println!("{what}: matches XAD to 1e-9 ({} values)", got.len());
}

/// Normals in the engine's layout: path block `b` (W paths), step `k`, draw `j`, lane `l`.
fn heston_normals<L: Lanes>(raw: &[f64]) -> Vec<f64> {
    let w = L::W;
    let mut z = vec![0.0; PATHS * STEPS * 2];
    for p in 0..PATHS {
        for j in 0..2 {
            for k in 0..STEPS {
                z[(((p / w) * STEPS + k) * 2 + j) * w + p % w] = raw[p * 2 * STEPS + j * STEPS + k];
            }
        }
    }
    z
}

fn heston<L: Lanes>(raw: &[f64], want: &[f64]) -> (f64, f64) {
    let z = heston_normals::<L>(raw);
    let p = HESTON.map(L::from);
    let r = paths::gradient::<Heston, L>(&p, STEPS, PATHS, &z);
    let got: Vec<f64> = std::iter::once(r.value).chain(r.grad).collect();
    check(&format!("heston W={}", L::W), &got, want);
    let primal = median_ms(3, 10, || {
        black_box(paths::value::<Heston, L>(black_box(&p), STEPS, PATHS, &z).value);
    });
    let grad = median_ms(3, 10, || {
        let r = paths::gradient::<Heston, L>(black_box(&p), STEPS, PATHS, &z);
        black_box((r.value, r.grad));
    });
    println!("raads heston W={:2}: primal {primal:.3} ms, gradient {grad:.3} ms", L::W);
    (primal, grad)
}

/// One optimizer iteration: objective and gradient (15) at `params`.
fn sabr_iter<L: Lanes<Elem = f64>>(s: &Sabr, params: &[f64; 15]) -> (f64, [f64; 15]) {
    let mut obj = 0.0;
    let mut g = [0.0; 15];
    for e in 0..EXPIRIES {
        let p = [params[3 * e], params[3 * e + 1], params[3 * e + 2]].map(L::from);
        let mut a = [L::from(0.0); 3];
        let mut v = L::from(0.0);
        for i in (e * STRIKES..(e + 1) * STRIKES).step_by(L::W) {
            let l = |x: &[f64]| L::load(&x[i..]);
            v += sabr_sq::vjp(&p, l(&s.f), l(&s.k), l(&s.t), l(&s.beta), l(&s.mkt), L::from(1.0), &mut a);
        }
        obj += v.sum();
        for j in 0..3 {
            g[3 * e + j] = a[j].sum();
        }
    }
    (obj, g)
}

/// Objective only.
fn sabr_obj<L: Lanes<Elem = f64>>(s: &Sabr, params: &[f64; 15]) -> f64 {
    let mut obj = 0.0;
    for e in 0..EXPIRIES {
        let p = [params[3 * e], params[3 * e + 1], params[3 * e + 2]].map(L::from);
        for i in (e * STRIKES..(e + 1) * STRIKES).step_by(L::W) {
            let l = |x: &[f64]| L::load(&x[i..]);
            obj += sabr_sq::eval(&p, l(&s.f), l(&s.k), l(&s.t), l(&s.beta), l(&s.mkt)).sum();
        }
    }
    obj
}

fn sabr<L: Lanes<Elem = f64>>(s: &Sabr, want0: &[f64], want_sum: &[f64]) -> (f64, f64) {
    let mut params = s.init;
    let (mut sum_obj, mut sum_g) = (0.0, [0.0; 15]);
    for (it, dp) in s.dp.iter().enumerate() {
        let (obj, g) = sabr_iter::<L>(s, &params);
        if it == 0 {
            check(&format!("sabr W={} iteration 0", L::W), &std::iter::once(obj).chain(g).collect::<Vec<_>>(), want0);
        }
        sum_obj += obj;
        for p in 0..15 {
            sum_g[p] += g[p];
            params[p] += dp[p];
        }
    }
    check(&format!("sabr W={} sums over 500 iterations", L::W), &std::iter::once(sum_obj).chain(sum_g).collect::<Vec<_>>(), want_sum);
    let run = |grad: bool| {
        median_ms(3, 10, || {
            let mut params = s.init;
            for dp in &s.dp {
                if grad {
                    black_box(sabr_iter::<L>(s, black_box(&params)));
                } else {
                    black_box(sabr_obj::<L>(s, black_box(&params)));
                }
                for p in 0..15 {
                    params[p] += dp[p];
                }
            }
        })
    };
    let (primal, grad) = (run(false), run(true));
    println!("raads sabr W={:2}: primal {primal:.3} ms, gradient {grad:.3} ms", L::W);
    (primal, grad)
}

/// `delta, lambda[80], L0[80]` (src/libor_swaption.hpp, `defaultMarket`).
fn libor_params<L: Lanes>() -> [L; 161] {
    std::array::from_fn(|i| {
        L::from(if i == 0 {
            0.05
        } else if i <= RATES {
            0.2
        } else {
            0.05
        })
    })
}

/// Normals per block of W paths: step `k`, lane `l`.
fn libor_normals<L: Lanes>(raw: &[f64]) -> Vec<[L; LSTEPS]> {
    raw.chunks(LSTEPS * L::W).map(|block| std::array::from_fn(|k| L::from_fn(|l| block[l * LSTEPS + k]))).collect()
}

fn libor_bench<L: Lanes<Elem = f64>>(raw: &[f64], want: &[f64]) -> (f64, f64) {
    let z = libor_normals::<L>(raw);
    let p = libor_params::<L>();
    let grad = |p: &[L; 161]| {
        let mut g = [L::from(0.0); 161];
        let mut v = L::from(0.0);
        for zb in &z {
            v += libor::vjp(p, zb, L::from(1.0), &mut g);
        }
        (v, g)
    };
    let (v, g) = grad(&p);
    let got: Vec<f64> = std::iter::once(v).chain(g).map(|x| x.sum() / PATHS as f64).collect();
    check(&format!("libor W={}", L::W), &got, want);
    let primal = median_ms(3, 10, || {
        let p = black_box(&p);
        black_box(z.iter().fold(L::from(0.0), |v, zb| v + libor::eval(p, zb)));
    });
    let grad = median_ms(3, 10, || {
        black_box(grad(black_box(&p)));
    });
    println!("raads libor W={:2}: primal {primal:.3} ms, gradient {grad:.3} ms", L::W);
    (primal, grad)
}

fn main() {
    let bytes = std::fs::read("target/adbench.bin").expect("run ad-benchmarks/reference.cpp first (benchmarks/ad-benchmarks/run.sh)");
    let all: Vec<f64> = bytes.as_chunks::<8>().0.iter().map(|b| f64::from_le_bytes(*b)).collect();
    let (raw, rest) = all.split_at(PATHS * 2 * STEPS);
    let mut s = Sabr { f: vec![], t: vec![], beta: vec![], k: vec![], mkt: vec![], init: [0.0; 15], dp: vec![] };
    let mut it = rest.iter().copied();
    for _ in 0..EXPIRIES {
        let (f, t, beta) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
        s.k.extend(it.by_ref().take(STRIKES));
        s.mkt.extend(it.by_ref().take(STRIKES));
        s.f.extend([f; STRIKES]);
        s.t.extend([t; STRIKES]);
        s.beta.extend([beta; STRIKES]);
    }
    s.init = std::array::from_fn(|_| it.next().unwrap());
    s.dp = (0..ITERS).map(|_| std::array::from_fn(|_| it.next().unwrap())).collect();
    let lz: Vec<f64> = it.by_ref().take(PATHS * LSTEPS).collect();
    assert!(it.next().is_none(), "unexpected trailing data");

    let refs = std::fs::read_to_string("target/adbench_reference.txt").expect("XAD reference values");
    let line = |name: &str| -> Vec<f64> {
        let l = refs.lines().find(|l| l.split_whitespace().next() == Some(name)).unwrap();
        l.split_whitespace().skip(1).map(|x| x.parse().unwrap()).collect()
    };
    let h = line("heston");
    let h1 = heston::<f64>(raw, &h);
    heston::<F64x2>(raw, &h);
    heston::<F64x4>(raw, &h);
    let h8 = heston::<F64x8>(raw, &h);
    heston::<F64x16>(raw, &h);
    let (s0, ssum) = (line("sabr0"), line("sabr_sum"));
    let s1 = sabr::<f64>(&s, &s0, &ssum);
    sabr::<F64x2>(&s, &s0, &ssum);
    let s4 = sabr::<F64x4>(&s, &s0, &ssum);
    let l = line("libor");
    let l1 = libor_bench::<f64>(&lz, &l);
    libor_bench::<F64x2>(&lz, &l);
    let l4 = libor_bench::<F64x4>(&lz, &l);
    libor_bench::<F64x8>(&lz, &l);
    let rows: String = [
        ("raads", "HestonMC", h1),
        ("raads SIMD", "HestonMC", h8),
        ("raads", "SABRCalib", s1),
        ("raads SIMD", "SABRCalib", s4),
        ("raads", "LiborSwaption", l1),
        ("raads SIMD", "LiborSwaption", l4),
    ]
    .iter()
    .map(|(lib, bench, (primal, grad))| format!("{lib},{bench},{primal:.6},{grad:.6}\n"))
    .collect();
    std::fs::write("target/adbench_raads.csv", rows).unwrap();
}
