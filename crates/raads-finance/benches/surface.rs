//! Full American-put chains: 2000 underliers x 20 maturities x 25 strikes = 1M options,
//! single core, synthetic market. One FD solve per underlier (unit-strike model on log(S/K)), f64:
//!   values  4 underliers per batch: price, delta, gamma, theta, dV/dK for every option
//!   Dual    2 underliers x (d/dsigma, d/dr) per batch: the above + vega and rho
//!   bumped  values + 4 bumped solves for vega and rho (for comparison)
//!   book    4 underliers per batch: book value + d/d(r, q, sigma) per underlier (one reverse sweep)
//! Every output is consumed. Pass `quick` to run h = 0.02 and 0.01 only.

use raads::lanes::{F64x4, F64x8};
use raads::{AdNum, Dual, Lanes, Real};
use raads_engines::pde1d::{self, Read, SnapshotGrid, Snapshots};
use raads_finance::bs_fd::BsPutMoneyness;
use raads_finance::{SurfaceOption, option};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use std::hint::black_box;
use std::time::Instant;

const NU: usize = 2000;
const NM: usize = 20;
const NK: usize = 25;
const RATE: f64 = 0.04;
type M = BsPutMoneyness<true>;

struct Market {
    s: Vec<f64>,
    sig: Vec<f64>,
    q: Vec<f64>,
    /// NU * NK, the same strikes for every maturity
    k: Vec<f64>,
    /// NU * NM * NK book positions
    pos: Vec<f64>,
    t: Vec<f64>,
    /// underliers sorted by sigma (similar grids per batch)
    order: Vec<usize>,
}

impl Market {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let mut n = || -> f64 { StandardNormal.sample(&mut rng) };
        let (mut s, mut sig, mut q, mut k) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..NU {
            s.push((80f64.ln() + n()).exp().clamp(5.0, 2000.0));
            sig.push((0.30f64.ln() + 0.40 * n()).exp().clamp(0.10, 1.20));
        }
        let mut rng = StdRng::seed_from_u64(43);
        for &sv in &s {
            q.push(if rng.random::<f64>() < 0.4 { 0.0 } else { rng.random_range(0.0..0.05) });
            let inc: f64 = if sv < 25.0 {
                0.5
            } else if sv < 100.0 {
                1.0
            } else if sv < 250.0 {
                2.5
            } else {
                5.0
            };
            for j in 0..NK {
                let m = 0.5 + j as f64 / (NK - 1) as f64;
                k.push(inc.max(((sv * m / inc + 0.5) as i64) as f64 * inc));
            }
        }
        let pos = (0..NU * NM * NK).map(|_| rng.random_range(-10.0..10.0)).collect();
        let mut t: Vec<f64> = (1..7).map(|w| w as f64 / 52.0).collect();
        t.extend([2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 24, 30, 36].map(|mo| mo as f64 / 12.0));
        let mut order: Vec<usize> = (0..NU).collect();
        order.sort_by(|&a, &b| sig[a].total_cmp(&sig[b]));
        Market { s, sig, q, k, pos, t, order }
    }

    /// x = log(S/K) range covering every strike of the batch ± 4.5 sd at the longest maturity.
    fn grid(&self, und: &[usize], h: f64, dt: f64) -> SnapshotGrid {
        let (mut lo, mut hi, mut smax) = (0.0f64, 0.0f64, 0.0f64);
        for &u in und {
            lo = lo.min((self.s[u] / self.k[u * NK + NK - 1]).ln());
            hi = hi.max((self.s[u] / self.k[u * NK]).ln());
            smax = smax.max(self.sig[u]);
        }
        let z = 4.5 * smax * self.t[NM - 1].sqrt();
        SnapshotGrid::new(lo - z, hi + z, ((hi - lo + 2.0 * z) / h).ceil() as usize + 1, dt)
    }

    fn lanes<L: Lanes>(&self, und: &[usize], f: &[f64]) -> L {
        L::from_fn(|l| f[und[l]])
    }

    fn strikes<L: Lanes>(&self, und: &[usize], j: usize) -> L {
        L::from_fn(|l| self.k[und[l] * NK + j])
    }
}

fn sum5<R: Real>(o: &SurfaceOption<R>) -> f64 {
    [o.price, o.delta, o.gamma, o.theta, o.dvdk].iter().map(|x| x.value().sum()).sum()
}

fn run_values<L: Lanes>(mk: &Market, h: f64, dt: f64) -> f64 {
    let w = L::W;
    let mut chk = 0.0;
    for und in mk.order.chunks_exact(w) {
        let p = [L::from(RATE), mk.lanes(und, &mk.q), mk.lanes(und, &mk.sig)];
        let s: Snapshots<L> = pde1d::snapshots::<M, L>(&p, &mk.grid(und, h, dt), &mk.t);
        let spot: L = mk.lanes(und, &mk.s);
        for m in 0..NM {
            for j in 0..NK {
                chk += sum5(&option(&s, m, spot, mk.strikes(und, j)));
            }
        }
    }
    chk
}

/// W lanes = W/2 underliers x (vega lane, rho lane).
fn run_dual<L: Lanes>(mk: &Market, h: f64, dt: f64) -> f64 {
    let w = L::W;
    let dsig = L::from_fn(|l| if l % 2 == 0 { 1.0 } else { 0.0 });
    let mut chk = 0.0;
    for b in 0..NU / (w / 2) {
        let und: Vec<usize> = (0..w).map(|l| mk.order[b * (w / 2) + l / 2]).collect();
        let p = [
            Dual::new(L::from(RATE), dsig.rsub(1.0)),
            Dual::new(mk.lanes(&und, &mk.q), L::from(0.0)),
            Dual::new(mk.lanes(&und, &mk.sig), dsig),
        ];
        let s = pde1d::snapshots::<M, Dual<L>>(&p, &mk.grid(&und, h, dt), &mk.t);
        let spot: L = mk.lanes(&und, &mk.s);
        for m in 0..NM {
            for j in 0..NK {
                let o = option(&s, m, spot, mk.strikes(&und, j));
                // even lanes: values + vega; odd lanes: rho (values repeated)
                chk += sum5(&o) + o.price.t.sum();
            }
        }
    }
    chk
}

fn run_bumped(mk: &Market, h: f64, dt: f64) -> f64 {
    type L = F64x4;
    let mut chk = 0.0;
    for und in mk.order.as_chunks::<{ L::W }>().0 {
        let g = mk.grid(und, h, dt);
        let sig: L = mk.lanes(und, &mk.sig);
        let surfs: Vec<Snapshots<L>> = [(0.0, 1.0), (0.0, 1.01), (0.0, 0.99), (1e-4, 1.0), (-1e-4, 1.0)]
            .iter()
            .map(|&(dr, fs)| pde1d::snapshots::<M, L>(&[L::from(RATE + dr), mk.lanes(und, &mk.q), sig * fs], &g, &mk.t))
            .collect();
        let spot: L = mk.lanes(und, &mk.s);
        for m in 0..NM {
            for j in 0..NK {
                let k: L = mk.strikes(und, j);
                let o = option(&surfs[0], m, spot, k);
                let pr: Vec<L> = surfs[1..].iter().map(|s| option(s, m, spot, k).price).collect();
                let vega = (pr[0] - pr[1]) / (sig * 0.02);
                let rho = (pr[2] - pr[3]) / 2e-4;
                chk += sum5(&o) + vega.sum() + rho.sum();
            }
        }
    }
    chk
}

fn run_book(mk: &Market, h: f64, dt: f64) -> f64 {
    type L = F64x4;
    let mut chk = 0.0;
    for und in mk.order.as_chunks::<{ L::W }>().0 {
        let p = [L::from(RATE), mk.lanes(und, &mk.q), mk.lanes(und, &mk.sig)];
        let spot: L = mk.lanes(und, &mk.s);
        let mut pts = Vec::with_capacity(NM * NK);
        for m in 0..NM {
            for j in 0..NK {
                let k: L = mk.strikes(und, j);
                let w = L::from_fn(|l| mk.pos[und[l] * NM * NK + m * NK + j]) * k;
                pts.push(Read { horizon: m, x: (spot / k).ln(), weight: w });
            }
        }
        let bk = pde1d::snapshots_weighted::<M, L>(&p, &mk.grid(und, h, dt), &mk.t, &pts);
        chk += bk.value.sum() + bk.grad.iter().map(|g| g.sum()).sum::<f64>();
    }
    chk
}

fn secs(f: impl FnOnce() -> f64) -> (f64, f64) {
    let t = Instant::now();
    let c = black_box(f());
    (t.elapsed().as_secs_f64(), c)
}

fn main() {
    let quick = std::env::args().any(|a| a == "quick");
    let mk = Market::new();
    println!("2000 underliers x 20 maturities x 25 strikes = 1M American puts; one FD solve per underlier, f64, single core");
    let cfgs: &[(f64, f64)] =
        if quick { &[(0.02, 1.0 / 100.0), (0.01, 1.0 / 250.0)] } else { &[(0.02, 1.0 / 100.0), (0.01, 1.0 / 250.0), (0.005, 1.0 / 500.0)] };
    for &(h, dt) in cfgs {
        let mut best = [f64::INFINITY; 4];
        let mut chk = 0.0;
        for _ in 0..3 {
            let runs: [&dyn Fn() -> f64; 4] =
                [&|| run_values::<F64x4>(&mk, h, dt), &|| run_dual::<F64x4>(&mk, h, dt), &|| run_bumped(&mk, h, dt), &|| {
                    run_book(&mk, h, dt)
                }];
            for (i, r) in runs.iter().enumerate() {
                let (s, c) = secs(r);
                best[i] = best[i].min(s);
                chk += c;
            }
        }
        println!(
            "h {h} dt {dt:.4} | values {:.3} s | + vega, rho via Dual {:.3} s ({:.2}x) | via bumps {:.3} s ({:.2}x) | book + 3 sensitivities {:.3} s ({:.2}x) | chk {chk:.6e}",
            best[0],
            best[1],
            best[1] / best[0],
            best[2],
            best[2] / best[0],
            best[3],
            best[3] / best[0]
        );
    }
    // SIMD width: wider batches give no extra throughput (h 0.01)
    let (h, dt) = (0.01, 1.0 / 250.0);
    let (v4, _) = secs(|| run_values::<F64x4>(&mk, h, dt));
    let (v8, _) = secs(|| run_values::<F64x8>(&mk, h, dt));
    let (d4, _) = secs(|| run_dual::<F64x4>(&mk, h, dt));
    let (d8, _) = secs(|| run_dual::<F64x8>(&mk, h, dt));
    println!("SIMD width, h 0.01: values W=4 {v4:.3} s | W=8 {v8:.3} s | Dual W=4 {d4:.3} s | Dual W=8 {d8:.3} s");
}
