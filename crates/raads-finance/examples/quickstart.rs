//! One American put by finite differences (price, Greeks, gamma via Dual, a strike ladder in
//! SIMD lanes) and an arbitrary function differentiated with `#[kernel]`.
//! Run: `cargo run --release -p raads-finance --example quickstart`

use raads::prelude::*;
use raads_engines::pde1d::{self, Grid};

// ---------- 1. one American put by finite differences ----------
pub struct AmericanPut;

// initial = payoff, x0 = log-spot, horizon = maturity, early exercise = the obstacle
#[kernels(raads_engines::pde1d)]
impl AmericanPut {
    const OBSTACLE: bool = true;

    // params p = [s0, r, sigma, T, K];  PDE in x = log S:  V_t = a V_xx + b V_x - c V
    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
        let (r, sigma) = (p[1], p[2]);
        [0.5 * sigma * sigma, r - 0.5 * sigma * sigma, r]
    }
    pub fn initial<N: AdNum>(#[data] x: N, p: [N; 5]) -> N {
        (p[4] - x.exp()).relu()
    }
    pub fn x0<N: AdNum>(p: [N; 5]) -> N {
        p[0].ln()
    }
    pub fn horizon<N: AdNum>(p: [N; 5]) -> N {
        p[3]
    }
}

// ---------- 2. any function, differentiated ----------
/// f(x, y; z) = exp(x·y) + sqrt(x² + z) / y
#[kernel]
pub fn f<N: AdNum>(v: [N; 2], #[data] z: N) -> N {
    let x = v[0];
    let y = v[1];
    (x * y).exp() + (x * x + z).sqrt() / y
}

fn main() {
    // --- FD, one option ---
    let p = [100.0, 0.05, 0.2, 1.0, 100.0]; // s0, r, sigma, T, K
    let grid = Grid::new(100f64.ln() - 1.5, 100f64.ln() + 1.5, 401, 400);
    let r = pde1d::gradient::<AmericanPut, f64>(&p, &grid);
    println!("American put {:.6}", r.value);
    println!("  delta {:.6}  rho {:.6}  vega {:.6}  dV/dT {:.6}  dV/dK {:.6}", r.grad[0], r.grad[1], r.grad[2], r.grad[3], r.grad[4]);

    // gamma etc.: seed a direction with Dual and read the tangent of the gradient (H·v)
    let pd = [Dual::new(100.0, 1.0), Dual::from(0.05), Dual::from(0.2), Dual::from(1.0), Dual::from(100.0)];
    let h = pde1d::gradient::<AmericanPut, Dual<f64>>(&pd, &grid);
    println!("  gamma {:.6}  vanna {:.6}", h.grad[0].t, h.grad[2].t);

    // four strikes at once in SIMD lanes, same grid
    let mut p4 = p.map(F64x4::from);
    p4[4] = F64x4::from_fn(|l| [90.0, 100.0, 110.0, 120.0][l]);
    let r4 = pde1d::gradient::<AmericanPut, F64x4>(&p4, &grid);
    println!("  K=90..120: {:?}", (0..4).map(|l| (r4.value.lane(l) * 1e4).round() / 1e4).collect::<Vec<_>>());

    // --- any function ---
    let y = f([0.5, 2.0], 1.0); // plain call
    let mut g = [0.0; 2];
    let y2 = f::vjp(&[0.5, 2.0], 1.0, 1.0, &mut g); // value + gradient
    println!("f = {y:.6} (vjp {y2:.6})  df/dx = {:.6}  df/dy = {:.6}", g[0], g[1]);

    // Hessian column: Dual seeded with e_x
    let mut hd = [Dual::from(0.0); 2];
    f::vjp(&[Dual::new(0.5, 1.0), Dual::from(2.0)], Dual::from(1.0), Dual::from(1.0), &mut hd);
    println!("d2f/dx2 = {:.6}  d2f/dydx = {:.6}", hd[0].t, hd[1].t);
}
