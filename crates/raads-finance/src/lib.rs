//! Option pricing on [`raads_engines`]: Black-Scholes and SVI models for the path and 1-D PDE
//! engines (price = the engine's value, Greeks = its gradient, second-order Greeks with
//! [`Dual`](raads::Dual) parameters) and option surfaces from one multi-maturity PDE solve.
//!
//! # Monte Carlo
//!
//! A Black-Scholes call in log space, priced with every Greek in one pass: the path
//! engine's `functional` is the discounted payoff. Lanes of the value type are paths; normals
//! are supplied by the caller (`f32`, laid out per [`paths::gradient`](raads_engines::paths::gradient)).
//!
//! ```
//! use raads::prelude::*;
//! use raads_engines::paths;
//!
//! pub struct GbmCall;
//!
//! // params: s0, r, sigma, T, K
//! #[kernels(raads_engines::paths)]
//! impl GbmCall {
//!     pub fn setup<N: AdNum>(p: [N; 5], #[data] n_steps: N) -> [N; 2] {
//!         let dt = p[3] / n_steps;
//!         [(p[1] - 0.5 * p[2] * p[2]) * dt, p[2] * dt.sqrt()]
//!     }
//!     pub fn init<N: AdNum>(p: [N; 5]) -> [N; 1] {
//!         [p[0].ln()]
//!     }
//!     pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
//!         [s[0] + c[0] + c[1] * z[0]]
//!     }
//!     pub fn functional<N: AdNum>(s: [N; 1], p: [N; 5], _c: [N; 2]) -> N {
//!         (s[0].exp() - p[4]).relu() * (-p[1] * p[3]).exp()
//!     }
//! }
//!
//! # fn normals(n: usize) -> Vec<f32> {
//! #     // LCG + Box-Muller: deterministic, no dependencies
//! #     let mut state = 0x2545_f491_4f6c_dd1d_u64;
//! #     let mut uniform = move || {
//! #         state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
//! #         ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
//! #     };
//! #     (0..n.div_ceil(2))
//! #         .flat_map(|_| {
//! #             let (r, t) = ((-2.0 * uniform().ln()).sqrt(), std::f64::consts::TAU * uniform());
//! #             [(r * t.cos()) as f32, (r * t.sin()) as f32]
//! #         })
//! #         .take(n)
//! #         .collect()
//! # }
//! let (n_paths, n_steps) = (65_536, 4);
//! let z = normals(n_paths * n_steps);
//! let params = [100.0, 0.01, 0.2, 1.0, 100.0].map(F64x4::from);
//! let r = paths::gradient::<GbmCall, F64x4>(&params, n_steps, n_paths, &z);
//! // Black-Scholes: price 8.433, delta 0.560
//! assert!((r.value - 8.433).abs() < 4.0 * r.std_err);
//! assert!((r.grad[0] - 0.560).abs() < 0.02);
//! ```
//!
//! # Finite differences
//!
//! An American put on a Crank–Nicolson grid in `x = log S`: the payoff is the PDE engine's
//! `initial` condition, `x0` the log-spot, `horizon` the maturity, and early exercise its
//! obstacle. Lanes of the value type are independent problems (here one, `f64`).
//!
//! ```
//! use raads::prelude::*;
//! use raads_engines::pde1d::{self, Grid};
//!
//! pub struct AmericanPut;
//!
//! // params: s0, r, sigma, T, K
//! #[kernels(raads_engines::pde1d)]
//! impl AmericanPut {
//!     const OBSTACLE: bool = true;
//!     pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 5]) -> [N; 3] {
//!         let (r, sigma) = (p[1], p[2]);
//!         [0.5 * sigma * sigma, r - 0.5 * sigma * sigma, r]
//!     }
//!     pub fn initial<N: AdNum>(#[data] x: N, p: [N; 5]) -> N {
//!         (p[4] - x.exp()).relu()
//!     }
//!     pub fn x0<N: AdNum>(p: [N; 5]) -> N {
//!         p[0].ln()
//!     }
//!     pub fn horizon<N: AdNum>(p: [N; 5]) -> N {
//!         p[3]
//!     }
//! }
//!
//! let grid = Grid::new(100f64.ln() - 1.2, 100f64.ln() + 1.2, 201, 200);
//! let r = pde1d::gradient::<AmericanPut, f64>(&[100.0, 0.01, 0.2, 1.0, 100.0], &grid);
//! assert!((r.value - 7.51).abs() < 0.02);
//! assert!(r.grad[0] < 0.0); // delta
//! ```

pub mod bs_fd;
pub mod gbm;
pub mod surface;
pub mod svi;

pub use surface::{SurfaceOption, option};
