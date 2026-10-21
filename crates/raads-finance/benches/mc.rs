//! MC engine: price + every Greek vs pricing only.
//! GBM call, log space, 200k scalar paths x 52 steps, single core, best of 31.
//! Every output is consumed.

use raads::lanes::{F32x16, F32x32};
use raads::{Arr, Lanes};
use raads_engines::paths::{self, PathModel};
use raads_finance::gbm::{GbmAsianCall, GbmCall};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::hint::black_box;
use std::time::Instant;

const N_STEPS: usize = 52;
const REPS: usize = 31;

fn best_ms(mut f: impl FnMut() -> f64) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..REPS {
        let t = Instant::now();
        chk += black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, chk)
}

fn run<M: PathModel, L: Lanes>(label: &str, n_paths: usize, z: &[f32]) {
    let w = L::W;
    let p: M::P<L> = Arr::from_fn(|i| L::from([100.0, 0.01, 0.2, 1.0, 100.0][i]));
    let (tp, _) = best_ms(|| paths::value::<M, L>(black_box(&p), N_STEPS, n_paths, z).value);
    let (tg, chk) = best_ms(|| {
        let r = paths::gradient::<M, L>(black_box(&p), N_STEPS, n_paths, z);
        r.value + r.grad.as_ref().iter().sum::<f64>()
    });
    println!("{label} W={w}: pricing {tp:.3} ms | price + 5 Greeks {tg:.3} ms ({:.2}x) | chk {chk:.4}", tg / tp);
}

fn main() {
    let n = 200_000 * N_STEPS;
    let mut rng = StdRng::seed_from_u64(7);
    let z: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    run::<GbmCall, F32x32>("GBM call f32", 200_000, &z);
    run::<GbmCall, F32x16>("GBM call f32", 200_000, &z);
    run::<GbmAsianCall, F32x32>("Asian call f32", 200_000, &z);
}
