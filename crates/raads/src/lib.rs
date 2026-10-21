//! Compile-time adjoint algorithmic differentiation.
//!
//! Write a branch-free function once, generic over [`AdNum`]; [`#[kernel]`](kernel) emits
//! forward and reverse code generic over [`Real`], straight-line apart from its own loops, so
//! one source gives the value, the gradient in one reverse pass, and with [`Dual`] values
//! Hessian-vector products (forward-over-reverse). The same code runs on `f64` and on SIMD packs ([`Lanes`]), one
//! independent evaluation per lane. [`#[kernels]`](kernels) compiles a set of kernels for an
//! engine that owns the runtime loops (e.g. the `raads-engines` crate).
//!
//! ```
//! use raads::prelude::*;
//!
//! /// f(x, y; z) = exp(x y) + sqrt(x² + z) / y
//! #[kernel]
//! pub fn f<N: AdNum>(v: [N; 2], #[data] z: N) -> N {
//!     let (x, y) = (v[0], v[1]);
//!     (x * y).exp() + (x * x + z).sqrt() / y
//! }
//!
//! // plain call
//! let value = f([0.5, 2.0], 1.0);
//!
//! // value and gradient [df/dx, df/dy] in one reverse pass
//! let mut grad = [0.0; 2];
//! assert_eq!(f::vjp(&[0.5, 2.0], 1.0, 1.0, &mut grad), value);
//! assert!((grad[0] - (2.0 * 1f64.exp() + 0.5 / 1.25f64.sqrt() / 2.0)).abs() < 1e-12);
//!
//! // Hessian column for x: the reverse pass on Dual values seeded with e_x
//! let mut h = [Dual::from(0.0); 2];
//! f::vjp(&[Dual::new(0.5, 1.0), Dual::from(2.0)], Dual::from(1.0), Dual::from(1.0), &mut h);
//! // d²f/dydx = (1 + xy) e^{xy} - x / (y² sqrt(x² + z))
//! assert!((h[1].t - (2.0 * 1f64.exp() - 0.125 / 1.25f64.sqrt())).abs() < 1e-12);
//!
//! // four evaluations at once, one per SIMD lane
//! let x = F64x4::from_fn(|l| l as f64);
//! let mut g4 = [F64x4::from(0.0); 2];
//! f::vjp(&[x, F64x4::from(2.0)], F64x4::from(1.0), F64x4::from(1.0), &mut g4);
//! ```
//!
//! # Toolchain and build flags
//!
//! raads needs nightly Rust: its SIMD packs are built on `std::simd` (`portable_simd`). The
//! repository pins the nightly in `rust-toolchain.toml`.
//!
//! The instructions are chosen at compile time from the enabled target features (no runtime
//! dispatch). aarch64 always has NEON. **On x86_64, build with `-C target-cpu=x86-64-v3`
//! (AVX2 + FMA), or `native` / `x86-64-v4` for AVX-512**; default x86_64 builds get 128-bit
//! SSE2 and no FMA (multiply and add are rounded separately), which is much slower.
//! For example in `.cargo/config.toml`:
//!
//! ```toml
//! [target.'cfg(target_arch = "x86_64")']
//! rustflags = ["-C", "target-cpu=x86-64-v3"]
//! ```

#![feature(portable_simd)]
#![warn(missing_docs)]

extern crate self as raads;

mod adnum;
mod arr;
pub mod lanes;
mod real;
mod trace;

pub use adnum::AdNum;
pub use arr::Arr;
pub use lanes::Lanes;
pub use raads_macros::{kernel, kernels};
pub use real::{Dual, DualVec, Real};

/// Everything a kernel needs: `use raads::prelude::*;`.
pub mod prelude {
    pub use crate::lanes::{F32x4, F32x8, F32x16, F32x32, F64x2, F64x4, F64x8, F64x16};
    pub use crate::{AdNum, Arr, Dual, DualVec, Lanes, Real, kernel, kernels};
}

/// Used by macro-generated code and tests; not public API.
#[doc(hidden)]
pub mod __private {
    pub use crate::trace::{Tracer, trace};
    pub use raads_codegen as codegen;
    pub use raads_macros::check_kernels;
}

mod sealed {
    /// Supertrait of [`AdNum`](crate::AdNum) (and so of [`Real`](crate::Real) and
    /// [`Lanes`](crate::Lanes)) and of [`Arr`](crate::Arr): only this crate implements them.
    pub trait Sealed {}
}
