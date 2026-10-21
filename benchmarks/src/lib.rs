//! Shared inputs and timing for the raads / Enzyme / XAD comparison. `xad/bench.cpp` mirrors
//! the generator and the workloads exactly, so every tool sees the same numbers.

use std::hint::black_box;
use std::time::Instant;

/// Monte Carlo: GBM European call, log-space exact scheme, price + 5 Greeks (s0, r, sigma, T, K).
pub const MC_PATHS: usize = 200_000;
pub const MC_STEPS: usize = 52;
/// Paths are stored in blocks of `MC_W` (the raads lane count): path `i * MC_W + l`, step `k`
/// at `(i * MC_STEPS + k) * MC_W + l`.
pub const MC_W: usize = 16;
pub const MC_PARAMS: [f64; 5] = [100.0, 0.01, 0.2, 1.0, 100.0];
/// General function f(x, y; z) = exp(x y) + sqrt(x² + z) / y, value and gradient at each point.
pub const F_POINTS: usize = 1 << 20;
/// Second order: H·v along `F_V` at every point.
pub const F_V: [f64; 2] = [1.0, 0.5];

/// SABR implied volatility (Hagan 2002), β = 0.5: gradient w.r.t. (F, α, ρ, ν) and H·v along
/// `SABR_V` at `F_POINTS` random (F, K, T, α, ρ, ν), strikes 5-45% out of the money.
pub const SABR_BETA: f64 = 0.5;
pub const SABR_V: [f64; 4] = [1.0, 0.5, -0.3, 0.2];

pub const REPS: usize = 31;

/// splitmix64.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }
}

/// `MC_PATHS * MC_STEPS` standard normals (Box-Muller), stored as `f32`.
pub fn normals() -> Vec<f32> {
    let mut rng = Rng::new(7);
    let mut z = Vec::with_capacity(MC_PATHS * MC_STEPS);
    while z.len() < MC_PATHS * MC_STEPS {
        let r = (-2.0 * rng.uniform().ln()).sqrt();
        let t = std::f64::consts::TAU * rng.uniform();
        z.push((r * t.cos()) as f32);
        z.push((r * t.sin()) as f32);
    }
    z
}

/// `F_POINTS` points: x in [0.1, 1.1), y in [1, 2), z in [0, 1).
pub fn points() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(11);
    let (mut x, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..F_POINTS {
        x.push(0.1 + rng.uniform());
        y.push(1.0 + rng.uniform());
        z.push(rng.uniform());
    }
    (x, y, z)
}

/// SABR inputs: `[F, α, ρ, ν]` per point (structure of arrays), and strikes and expiries.
pub struct SabrPoints {
    pub p: [Vec<f64>; 4],
    pub k: Vec<f64>,
    pub t: Vec<f64>,
}

pub fn sabr_points() -> SabrPoints {
    let mut rng = Rng::new(13);
    let mut s = SabrPoints { p: Default::default(), k: Vec::new(), t: Vec::new() };
    for _ in 0..F_POINTS {
        let f = 80.0 + 20.0 * rng.uniform();
        s.p[0].push(f);
        s.p[1].push(1.0 + 2.0 * rng.uniform());
        s.p[2].push(-0.6 + 0.8 * rng.uniform());
        s.p[3].push(0.2 + 0.6 * rng.uniform());
        s.k.push(f * (1.05 + 0.4 * rng.uniform()));
        s.t.push(0.25 + 2.0 * rng.uniform());
    }
    s
}

/// Best wall time in ms over `REPS` runs, and the sum of the checksums `f` returns.
pub fn best_ms(mut f: impl FnMut() -> f64) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..REPS {
        let t = Instant::now();
        chk += black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, chk)
}

pub fn report_mc(tool: &str, price: f64, g: &[f64; 5]) {
    println!("{tool:8} mc  price {price:.10} grad [{:.10}, {:.10}, {:.10}, {:.10}, {:.10}]", g[0], g[1], g[2], g[3], g[4]);
}

pub fn report_f(tool: &str, v: &[f64], gx: &[f64], gy: &[f64]) {
    let s = |a: &[f64]| a.iter().sum::<f64>();
    println!("{tool:8} f   sum f {:.10} sum df/dx {:.10} sum df/dy {:.10}", s(v), s(gx), s(gy));
}

/// Gradient and H·v at every point: `[df/dx, df/dy, (Hv)_x, (Hv)_y]`.
pub fn report_f_hv(tool: &str, h: &[Vec<f64>; 4]) {
    let s = |a: &[f64]| a.iter().sum::<f64>();
    println!("{tool:8} f2  sum grad [{:.10}, {:.10}] sum H·v [{:.10}, {:.10}]", s(&h[0]), s(&h[1]), s(&h[2]), s(&h[3]));
}

/// Sums of the value, gradient and H·v over all SABR points.
pub fn report_sabr(tool: &str, h: &[Vec<f64>; 9]) {
    let s: Vec<f64> = h.iter().map(|a| a.iter().sum()).collect();
    println!("{tool:8} sabr sum vol {:.10} grad {:.10?} H·v {:.10?}", s[0], &s[1..5], &s[5..]);
}

pub fn report_time(tool: &str, what: &str, ms: f64) {
    println!("{tool:8} {what:32} {ms:9.3} ms");
}
