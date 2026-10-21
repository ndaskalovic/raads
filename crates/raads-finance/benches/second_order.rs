//! Second-order cost vs the gradient, MC and FD, 5 parameters. Best of 11, single core.
//!   H·v            one forward-over-reverse pass (Dual parameters seeded with v)
//!   AD Hessian     5 such passes (v = e_j)
//!   1-pass Hessian one pass on DualVec<_, 5> (all 5 directions at once)
//!   bumped Hessian 11 gradient passes (central differences of the AAD gradient)
//! Every output is consumed.

use raads::lanes::{F32x8, F32x16, F32x32, F64x4, F64x8};
use raads::{Arr, Dual, DualVec, Lanes};
use raads_engines::paths::{self, PathModel};
use raads_engines::pde1d::{self, Grid, PdeModel};
use raads_finance::bs_fd::{BsAmericanPut, BsAmericanPutSmooth};
use raads_finance::gbm::GbmCallSmooth;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::hint::black_box;
use std::time::Instant;

const NS: usize = 52;
const REPS: usize = 11;
const P: [f64; 5] = [100.0, 0.01, 0.2, 1.0, 100.0];
const V: [f64; 5] = [1.0, 0.3, -0.5, 0.2, 0.7];

fn bumped(j: usize, s: f64) -> [f64; 5] {
    let mut q = P;
    q[j] += s * 1e-2 * P[j].abs().max(1e-2);
    q
}

fn timed(t: &mut f64, f: impl FnOnce() -> f64) -> f64 {
    let s = Instant::now();
    let c = black_box(f());
    *t = t.min(s.elapsed().as_secs_f64() * 1e3);
    c
}

fn mc<L: Lanes>(z: &[f32]) {
    type M = GbmCallSmooth;
    let n_paths = 200_000;
    let vec = |p: [f64; 5]| -> <M as PathModel>::P<L> { Arr::from_fn(|i| L::from(p[i])) };
    let dual = |v: [f64; 5]| -> <M as PathModel>::P<Dual<L>> { Arr::from_fn(|i| Dual::new(L::from(P[i]), L::from(v[i]))) };
    let unit = |j: usize| -> [f64; 5] { std::array::from_fn(|i| if i == j { 1.0 } else { 0.0 }) };
    let units = || -> <M as PathModel>::P<DualVec<L, 5>> { Arr::from_fn(|i| DualVec::unit(L::from(P[i]), i)) };
    let mut b = [f64::INFINITY; 5];
    let mut chk = 0.0;
    for _ in 0..REPS {
        chk += timed(&mut b[4], || {
            let h = paths::gradient::<M, DualVec<L, 5>>(&units(), NS, n_paths, z);
            h.grad_tangent.iter().flatten().sum::<f64>()
        });
        chk += timed(&mut b[0], || {
            let g = paths::gradient::<M, L>(&vec(P), NS, n_paths, z);
            g.value + g.grad.iter().sum::<f64>()
        });
        chk += timed(&mut b[1], || {
            let h = paths::gradient::<M, Dual<L>>(&dual(V), NS, n_paths, z);
            h.value_tangent[0] + h.grad_tangent[0].iter().sum::<f64>()
        });
        chk += timed(&mut b[2], || {
            (0..5).map(|j| paths::gradient::<M, Dual<L>>(&dual(unit(j)), NS, n_paths, z).grad_tangent[0].iter().sum::<f64>()).sum()
        });
        chk += timed(&mut b[3], || {
            let base = paths::gradient::<M, L>(&vec(P), NS, n_paths, z).value;
            base + (0..5)
                .map(|j| {
                    let (gp, gm) = (
                        paths::gradient::<M, L>(&vec(bumped(j, 1.0)), NS, n_paths, z),
                        paths::gradient::<M, L>(&vec(bumped(j, -1.0)), NS, n_paths, z),
                    );
                    (0..5).map(|i| gp.grad[i] - gm.grad[i]).sum::<f64>()
                })
                .sum::<f64>()
        });
    }
    println!(
        "MC smooth call 200k x {NS} {} W={} | grad {:.3} ms | H.v {:.3} ({:.2}x) | AD Hessian {:.3} ({:.1}x) | 1-pass Hessian {:.3} ({:.1}x) | bumped Hessian {:.3} ({:.1}x) | chk {chk:.4}",
        std::any::type_name::<L::Elem>(),
        L::W,
        b[0],
        b[1],
        b[1] / b[0],
        b[2],
        b[2] / b[0],
        b[4],
        b[4] / b[0],
        b[3],
        b[3] / b[0]
    );
}

fn fd<M: PdeModel, L: Lanes>(nx: usize, nt: usize) {
    let g = Grid::new(100f64.ln() - 1.2, 100f64.ln() + 1.2, nx, nt);
    let vp = |q: [f64; 5]| -> M::P<L> { Arr::from_fn(|i| L::from(q[i])) };
    let dp = |t: [f64; 5]| -> M::P<Dual<L>> { Arr::from_fn(|i| Dual::new(L::from(P[i]), L::from(t[i]))) };
    let v: [f64; 5] = std::array::from_fn(|j| 0.3 + 0.1 * j as f64);
    let s = |x: L| x.sum();
    let units = || -> M::P<DualVec<L, 5>> { Arr::from_fn(|i| DualVec::unit(L::from(P[i]), i)) };
    let mut b = [f64::INFINITY; 6];
    let mut chk = 0.0;
    for _ in 0..REPS {
        chk += timed(&mut b[5], || {
            let r = pde1d::gradient::<M, DualVec<L, 5>>(&units(), &g);
            r.grad.as_ref().iter().flat_map(|x| x.t).map(s).sum::<f64>()
        });
        chk += timed(&mut b[0], || s(pde1d::value::<M, L>(&vp(P), &g)));
        chk += timed(&mut b[1], || {
            let r = pde1d::gradient::<M, L>(&vp(P), &g);
            s(r.value) + r.grad.as_ref().iter().map(|&x| s(x)).sum::<f64>()
        });
        chk += timed(&mut b[2], || {
            let r = pde1d::gradient::<M, Dual<L>>(&dp(v), &g);
            s(r.value.t) + r.grad.as_ref().iter().map(|x| s(x.t)).sum::<f64>()
        });
        chk += timed(&mut b[3], || {
            (0..5)
                .map(|j| {
                    let e: [f64; 5] = std::array::from_fn(|i| if i == j { 1.0 } else { 0.0 });
                    pde1d::gradient::<M, Dual<L>>(&dp(e), &g).grad.as_ref().iter().map(|x| s(x.t)).sum::<f64>()
                })
                .sum()
        });
        chk += timed(&mut b[4], || {
            let base = s(pde1d::gradient::<M, L>(&vp(P), &g).value);
            base + (0..5)
                .map(|j| {
                    let (gp, gm) = (pde1d::gradient::<M, L>(&vp(bumped(j, 1.0)), &g), pde1d::gradient::<M, L>(&vp(bumped(j, -1.0)), &g));
                    (0..5).map(|i| s(gp.grad.as_ref()[i] - gm.grad.as_ref()[i])).sum::<f64>()
                })
                .sum::<f64>()
        });
    }
    println!(
        "FD American put eps {} {nx}x{nt} W={} | price {:.3} ms | grad {:.3} ({:.2}x price) | H.v {:.3} ({:.2}x grad) | AD Hessian {:.3} ({:.1}x grad) | 1-pass Hessian {:.3} ({:.1}x grad) | bumped {:.3} ({:.1}x grad) | chk {chk:.3}",
        M::OBSTACLE_EPS,
        L::W,
        b[0],
        b[1],
        b[1] / b[0],
        b[2],
        b[2] / b[1],
        b[3],
        b[3] / b[1],
        b[5],
        b[5] / b[1],
        b[4],
        b[4] / b[1]
    );
}

fn main() {
    let mut rng = StdRng::seed_from_u64(7);
    let z: Vec<f32> = (0..200_000 * NS).map(|_| StandardNormal.sample(&mut rng)).collect();
    mc::<F32x8>(&z); // DualVec<_, 5> is fastest on narrow packs (fewer spills)
    mc::<F32x16>(&z);
    mc::<F32x32>(&z);
    mc::<F64x4>(&z);
    mc::<F64x8>(&z);
    fd::<BsAmericanPut, f64>(201, 200);
    fd::<BsAmericanPut, f64>(401, 400);
    fd::<BsAmericanPut, F64x4>(401, 400);
    fd::<BsAmericanPutSmooth, f64>(201, 200);
    fd::<BsAmericanPutSmooth, f64>(401, 400);
    fd::<BsAmericanPutSmooth, F64x4>(401, 400);
}
