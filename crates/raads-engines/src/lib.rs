//! Domain-neutral engines on the [`raads`] core: models are a few small kernels marked
//! `#[kernels(engine)]`; the engines own the runtime loops and return values with gradients
//! w.r.t. every model parameter (and Hessian-vector products with [`Dual`](raads::Dual)
//! parameters).
//!
//! - [`paths`]: Monte Carlo means of path functionals (pathwise adjoints, lanes = paths).
//! - [`pde1d`]: 1-D parabolic PDEs by the theta scheme, with an optional obstacle constraint
//!   (lanes = independent problems).

#![warn(missing_docs)]

pub mod paths;
pub mod pde1d;

/// Compiles and runs the README examples.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
struct ReadmeDoctest;
