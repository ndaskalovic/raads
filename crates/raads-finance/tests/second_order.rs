//! Second-order Greeks by forward-over-reverse. Asserted:
//! - every AD Hessian equals central differences of the exact AAD gradient (it is the exact
//!   second derivative of the discrete estimator / scheme) and is symmetric;
//! - FD European call: all entries within 2e-3 of the Black-Scholes Hessian except d²V/dK²
//!   (the strike only enters the kinked payoff), which bumping the AAD gradient gets within 5e-3;
//! - FD American put: with smoothed exercise every entry except d²V/dK² matches a 1%-bump
//!   reference within 5e-3; with the hard max only the S0 row (gamma, vanna, ...) is valid;
//! - MC (softplus payoff, eps = 0.5): gamma and vanna within 4 standard errors of Black-Scholes;
//!   the kinked payoff needs bumping of the AAD gradient.

use raads::lanes::F64x8;
use raads::{Arr, Dual};
use raads_engines::paths::{self, PathModel};
use raads_engines::pde1d::{self, Grid, PdeModel};
use raads_finance::bs_fd::{BsAmericanPut, BsAmericanPutSmooth, BsEuroCall};
use raads_finance::gbm::{GbmCall, GbmCallSmooth};

mod common;
use common::{Checks, rel};

const NP: usize = 5;
const S0: usize = 0;
const SIG: usize = 2;
const STRIKE: usize = 4;
const P: [f64; NP] = [100.0, 0.01, 0.2, 1.0, 100.0];
const N_STEPS: usize = 52;
const N_PATHS: usize = 200_000;

/// Black-Scholes Hessian over (S0, r, sigma, T, K), mpmath at 40 digits, row-major.
const BS: [f64; 25] = [
    0.019723966545394445,
    1.9723966545394445,
    0.09861983272697222,
    0.029585949818091667,
    -0.019723966545394445,
    1.9723966545394445,
    149.7112149070298,
    -29.585949818091667,
    46.066967714175774,
    -1.497112149070298,
    0.09861983272697222,
    -29.585949818091667,
    -1.4792974909045834,
    19.28017729812307,
    0.29585949818091667,
    0.029585949818091667,
    46.066967714175774,
    19.28017729812307,
    -2.0313904075939707,
    0.014614828327388687,
    -0.019723966545394445,
    -1.497112149070298,
    0.29585949818091667,
    0.014614828327388687,
    0.019723966545394445,
];

/// Price and row-major Hessian.
struct Hessian {
    price: f64,
    h: [f64; NP * NP],
}

impl Hessian {
    fn get(&self, i: usize, j: usize) -> f64 {
        self.h[i * NP + j]
    }
}

fn bump(p: [f64; NP], j: usize, rel: f64, sign: f64) -> [f64; NP] {
    let mut q = p;
    q[j] += sign * rel * p[j].abs().max(1e-2);
    q
}

fn grid() -> Grid {
    Grid::new(100f64.ln() - 1.2, 100f64.ln() + 1.2, 401, 400)
}

fn normals() -> Vec<f32> {
    common::normals(N_PATHS * N_STEPS, 42)
}

/// One forward-over-reverse pass per column (Dual parameters seeded with e_j).
fn fd_hessian<M: PdeModel>(p: [f64; NP]) -> Hessian {
    let mut h = [0.0; NP * NP];
    let mut price = 0.0;
    for j in 0..NP {
        let pd: M::P<Dual<f64>> = Arr::from_fn(|i| Dual::new(p[i], if i == j { 1.0 } else { 0.0 }));
        let r = pde1d::gradient::<M, Dual<f64>>(&pd, &grid());
        price = r.value.v;
        for i in 0..NP {
            h[i * NP + j] = r.grad.as_ref()[i].t;
        }
    }
    Hessian { price, h }
}

fn mc_hessian<M: PathModel>(p: [f64; NP], z: &[f32]) -> Hessian {
    type D = Dual<F64x8>;
    let mut h = [0.0; NP * NP];
    let mut price = 0.0;
    for j in 0..NP {
        let pd: M::P<D> = Arr::from_fn(|i| D::new(F64x8::from(p[i]), F64x8::from(if i == j { 1.0 } else { 0.0 })));
        let r = paths::gradient::<M, D>(&pd, N_STEPS, N_PATHS, z);
        price = r.value;
        for i in 0..NP {
            h[i * NP + j] = r.grad_tangent[0].as_ref()[i];
        }
    }
    Hessian { price, h }
}

/// Reference (not AD): central differences of the exact first-order AAD gradient.
fn fd_hessian_bumped<M: PdeModel>(p: [f64; NP], rel: f64) -> Hessian {
    let g = |q: [f64; NP]| pde1d::gradient::<M, f64>(&Arr::from_fn(|i| q[i]), &grid());
    let mut h = [0.0; NP * NP];
    for j in 0..NP {
        let e = rel * p[j].abs().max(1e-2);
        let (gp, gm) = (g(bump(p, j, rel, 1.0)), g(bump(p, j, rel, -1.0)));
        for i in 0..NP {
            h[i * NP + j] = (gp.grad.as_ref()[i] - gm.grad.as_ref()[i]) / (2.0 * e);
        }
    }
    Hessian { price: g(p).value, h }
}

fn mc_hessian_bumped<M: PathModel>(p: [f64; NP], z: &[f32], rel: f64) -> Hessian {
    let g = |q: [f64; NP]| paths::gradient::<M, F64x8>(&Arr::from_fn(|i| F64x8::from(q[i])), N_STEPS, N_PATHS, z);
    let mut h = [0.0; NP * NP];
    for j in 0..NP {
        let e = rel * p[j].abs().max(1e-2);
        let (gp, gm) = (g(bump(p, j, rel, 1.0)), g(bump(p, j, rel, -1.0)));
        for i in 0..NP {
            h[i * NP + j] = (gp.grad.as_ref()[i] - gm.grad.as_ref()[i]) / (2.0 * e);
        }
    }
    Hessian { price: g(p).value, h }
}

fn max_asym(h: &Hessian) -> f64 {
    let mut m = 0.0f64;
    for i in 0..NP {
        for j in i + 1..NP {
            m = m.max((h.get(i, j) - h.get(j, i)).abs() / (h.get(i, j).abs() + h.get(j, i).abs() + 1e-12));
        }
    }
    m
}

fn max_rel_vs(a: &Hessian, b: &Hessian, floor: f64, skip_kk: bool) -> f64 {
    let mut m = 0.0f64;
    for i in 0..NP {
        for j in 0..NP {
            if !(skip_kk && i == STRIKE && j == STRIKE) {
                m = m.max((a.get(i, j) - b.get(i, j)).abs() / b.get(i, j).abs().max(floor));
            }
        }
    }
    m
}

#[test]
fn fd_european() {
    let mut c = Checks::default();
    let he = fd_hessian::<BsEuroCall>(P);
    let tiny = fd_hessian_bumped::<BsEuroCall>(P, 1e-4);
    let big = fd_hessian_bumped::<BsEuroCall>(P, 1e-2);
    c.le("exact: max rel vs FD of AAD gradient (h=1e-4)", max_rel_vs(&he, &tiny, 1e-6, false), 1e-5);
    c.le("symmetry: max rel asymmetry", max_asym(&he), 1e-9);
    let mut worst = 0.0f64;
    for i in 0..NP {
        for j in 0..NP {
            if !(i == STRIKE && j == STRIKE) {
                worst = worst.max(rel(he.get(i, j), BS[i * NP + j]));
            }
        }
    }
    c.le("vs Black-Scholes, max rel (all but d2/dK2)", worst, 2e-3);
    c.rel("gamma", he.get(S0, S0), BS[0], 2e-3);
    c.rel("vanna", he.get(S0, SIG), BS[2], 2e-3);
    c.rel("volga", he.get(SIG, SIG), BS[12], 2e-3);
    c.rel("d2/dK2 by bumping the AAD gradient (1%)", big.get(STRIKE, STRIKE), BS[24], 5e-3);
    println!("    AD d2/dK2 = {} (the strike only enters through the kinked payoff)", he.get(STRIKE, STRIKE));
    c.assert_ok();
}

#[test]
fn fd_american() {
    let mut c = Checks::default();
    let ha = fd_hessian::<BsAmericanPutSmooth>(P);
    let ha_tiny = fd_hessian_bumped::<BsAmericanPutSmooth>(P, 1e-4);
    let ha_big = fd_hessian_bumped::<BsAmericanPutSmooth>(P, 1e-2);
    c.le("smoothed: exact vs FD of AAD gradient (h=1e-4)", max_rel_vs(&ha, &ha_tiny, 1e-3, false), 2e-3);
    c.le("smoothed: symmetry", max_asym(&ha), 1e-9);
    c.le("smoothed: vs 1%-bump reference (all but d2/dK2)", max_rel_vs(&ha, &ha_big, 1e-3, true), 5e-3);
    let hard = fd_hessian::<BsAmericanPut>(P);
    let hard_big = fd_hessian_bumped::<BsAmericanPut>(P, 1e-2);
    let hw = (0..NP).map(|j| rel(hard.get(S0, j), hard_big.get(S0, j))).fold(0.0, f64::max);
    c.le("hard max: S0 row vs 1%-bump reference", hw, 2e-3);
    c.rel("hard max gamma vs smoothed gamma", hard.get(S0, S0), ha.get(S0, S0), 2e-3);
    c.rel("d2/dK2 by bumping (hard max vs smoothed)", hard_big.get(STRIKE, STRIKE), ha_big.get(STRIKE, STRIKE), 1e-2);
    println!("    price: hard max {}, smoothed {} (use the hard max for price / first order)", hard.price, ha.price);
    c.assert_ok();
}

#[test]
fn mc() {
    let mut c = Checks::default();
    let z = normals();
    let hm = mc_hessian::<GbmCallSmooth>(P, &z);
    let hm_tiny = mc_hessian_bumped::<GbmCallSmooth>(P, &z, 1e-4);
    c.le("exact: max rel vs FD of AAD gradient (h=1e-4)", max_rel_vs(&hm, &hm_tiny, 1e-6, false), 1e-5);
    c.le("symmetry", max_asym(&hm), 1e-9);
    // 4 single-run standard errors (measured over 8 seeds): gamma 1.4e-4, vanna 3.1e-3
    c.abs("gamma vs Black-Scholes (4 SE)", hm.get(S0, S0), BS[0], 6e-4);
    c.abs("vanna vs Black-Scholes (4 SE)", hm.get(S0, SIG), BS[2], 1.3e-2);
    let hk = mc_hessian::<GbmCall>(P, &z);
    let hkb = mc_hessian_bumped::<GbmCall>(P, &z, 1e-2);
    println!("    kinked payoff: AD (pathwise) gamma {} <- misses the Dirac term at the strike", hk.get(S0, S0));
    c.abs("kinked payoff: gamma by bumping (1%)", hkb.get(S0, S0), BS[0], 1.5e-3);
    println!("    price {}", hm.price);
    c.assert_ok();
}
