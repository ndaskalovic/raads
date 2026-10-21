//! FD engine: grids and SIMD widths, price only vs price + 5 Greeks. American put, single core,
//! best of 31.
//! Every output is consumed.

use raads::lanes::{F32x8, F64x4};
use raads::{Lanes, Real};
use raads_engines::pde1d::{self, Grid};
use raads_finance::bs_fd::BsAmericanPut;
use std::hint::black_box;
use std::time::Instant;

fn best_ms(reps: usize, mut f: impl FnMut() -> f64) -> f64 {
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..reps {
        let t = Instant::now();
        chk += black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    black_box(chk);
    best
}

fn run<R: Real>(label: &str, nx: usize, nt: usize) {
    let strikes = [90.0, 100.0, 110.0, 120.0, 95.0, 105.0, 115.0, 125.0];
    let mut p = [100.0, 0.01, 0.2, 1.0, 100.0].map(R::from);
    p[4] = R::lift(R::Lanes::from_fn(|l| strikes[l % 8]));
    let g = Grid::new(100f64.ln() - 1.2, 100f64.ln() + 1.2, nx, nt);
    let sum = |v: R| v.value().sum();
    let tp = best_ms(31, || sum(pde1d::value::<BsAmericanPut, R>(black_box(&p), &g)));
    let tg = best_ms(31, || {
        let r = pde1d::gradient::<BsAmericanPut, R>(black_box(&p), &g);
        sum(r.value) + r.grad.iter().map(|&x| sum(x)).sum::<f64>()
    });
    println!("{label} W={} {nx}x{nt}: price {tp:.3} ms | price + 5 Greeks {tg:.3} ms ({:.2}x)", R::Lanes::W, tg / tp);
}

fn main() {
    run::<f64>("f64", 201, 200);
    run::<f64>("f64", 401, 400);
    run::<f64>("f64", 801, 800);
    run::<F64x4>("f64", 201, 200);
    run::<F64x4>("f64", 401, 400);
    run::<F32x8>("f32", 201, 200);
}
