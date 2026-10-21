//! Path engine: Monte Carlo mean of a path functional and its gradient w.r.t. every model
//! parameter (pathwise adjoints).
//!
//! A model is four small kernels written once against [`AdNum`](raads::AdNum) inside an
//! `impl` block marked `#[kernels(raads_engines::paths)]`:
//!
//! ```text
//! setup(p, #[data] n_steps) -> c      once per run    (per-step coefficients)
//! init(p)                   -> s0     once per run    (initial state)
//! step(s, c, #[data] z)     -> s'     every step      (z: NZ normal draws)
//! functional(s, p, c)       -> value  every path
//! ```
//!
//! Per SIMD path the engine runs the steps forward (saving only the residuals the step's
//! reverse sweep reads), the functional fused with its reverse sweep, and the steps backward;
//! the reverse of `init` and `setup` runs once per block of paths. Lanes are paths.
//!
//! ```
//! use raads::prelude::*;
//! use raads_engines::paths;
//!
//! /// Brownian motion with drift: X_0 = x0, X_T = x0 + mu T + sigma W_T; value X_T².
//! /// params: x0, mu, sigma, T.
//! pub struct Drift;
//!
//! #[kernels(raads_engines::paths)]
//! impl Drift {
//!     pub fn setup<N: AdNum>(p: [N; 4], #[data] n_steps: N) -> [N; 2] {
//!         let dt = p[3] / n_steps;
//!         [p[1] * dt, p[2] * dt.sqrt()]
//!     }
//!     pub fn init<N: AdNum>(p: [N; 4]) -> [N; 1] {
//!         [p[0]]
//!     }
//!     pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
//!         [s[0] + c[0] + c[1] * z[0]]
//!     }
//!     pub fn functional<N: AdNum>(s: [N; 1], _p: [N; 4], _c: [N; 2]) -> N {
//!         s[0] * s[0]
//!     }
//! }
//!
//! // antithetic pairs of ±1 draws in place of normals: X_T = ±1 for every path
//! let (n_steps, n_paths) = (1, 8);
//! let z: Vec<f32> = (0..n_paths).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
//! let r = paths::gradient::<Drift, F64x4>(&[0.0, 0.0, 1.0, 1.0].map(F64x4::from), n_steps, n_paths, &z);
//! assert_eq!(r.value, 1.0); // E[X_T²] = sigma² T
//! assert_eq!(r.grad[2], 2.0); // d/dsigma
//! ```

use raads::{Arr, Lanes, Real};

/// A model compiled by `#[kernels(raads_engines::paths)]`. The associated array types fix the
/// sizes: parameters `P`, per-step coefficients `C`, state `S`, normals per step `Z`.
///
/// Implement it with `#[kernels]`, not by hand: the methods are generated (hidden from these
/// docs) and may change between minor versions.
pub trait PathModel {
    /// Parameters.
    type P<T: Copy>: Arr<T>;
    /// Per-step coefficients.
    type C<T: Copy>: Arr<T>;
    /// Path state.
    type S<T: Copy>: Arr<T>;
    /// Normal draws per step.
    type Z<T: Copy>: Arr<T>;

    // Generated. `*_eval` compute a kernel; `*_vjp` add the adjoints of its inputs (seeded with
    // `a_y`) into the `a_*` arguments; `step_fwd` / `step_bwd` split the step's vector-Jacobian
    // product around the saved residuals `res`.
    /// Residuals `step_fwd` saves per step.
    #[doc(hidden)]
    const STEP_NRES: usize;

    /// `setup` kernel.
    #[doc(hidden)]
    fn setup_eval<R: Real>(p: &Self::P<R>, n_steps: R) -> Self::C<R>;
    /// `setup` reverse.
    #[doc(hidden)]
    fn setup_vjp<R: Real>(p: &Self::P<R>, n_steps: R, a_y: &Self::C<R>, a_p: &mut Self::P<R>) -> Self::C<R>;
    /// `init` kernel.
    #[doc(hidden)]
    fn init_eval<R: Real>(p: &Self::P<R>) -> Self::S<R>;
    /// `init` reverse.
    #[doc(hidden)]
    fn init_vjp<R: Real>(p: &Self::P<R>, a_y: &Self::S<R>, a_p: &mut Self::P<R>) -> Self::S<R>;
    /// `step` kernel.
    #[doc(hidden)]
    fn step_eval<R: Real>(s: &Self::S<R>, c: &Self::C<R>, z: &Self::Z<R>) -> Self::S<R>;
    /// `step` kernel, saving `STEP_NRES` residuals.
    #[doc(hidden)]
    fn step_fwd<R: Real>(s: &Self::S<R>, c: &Self::C<R>, z: &Self::Z<R>, res: &mut [R]) -> Self::S<R>;
    /// `step` reverse from the residuals.
    #[doc(hidden)]
    fn step_bwd<R: Real>(z: &Self::Z<R>, res: &[R], a_y: &Self::S<R>, a_s: &mut Self::S<R>, a_c: &mut Self::C<R>);
    /// `functional` kernel.
    #[doc(hidden)]
    fn functional_eval<R: Real>(s: &Self::S<R>, p: &Self::P<R>, c: &Self::C<R>) -> R;
    /// `functional` and its reverse in one pass.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn functional_vjp<R: Real>(
        s: &Self::S<R>,
        p: &Self::P<R>,
        c: &Self::C<R>,
        a_y: R,
        a_s: &mut Self::S<R>,
        a_p: &mut Self::P<R>,
        a_c: &mut Self::C<R>,
    ) -> R;
}

/// Implements [`PathModel`] for `#[kernels(raads_engines::paths)]`; see `raads::kernels`.
#[doc(hidden)]
#[macro_export]
macro_rules! __raads_paths_model {
    (@impl [$($g:tt)*] [$($ty:tt)*] [$($wh:tt)*] [$($c:tt)*] [$np:literal $nc:literal $ns:literal $nz:literal] [
        setup $_n0:literal [$($se:tt)*] [$($sv:tt)*] [$($_sf:tt)*] [$($_sb:tt)*]
        init $_n1:literal [$($ie:tt)*] [$($iv:tt)*] [$($_if:tt)*] [$($_ib:tt)*]
        step $nres:literal [$($te:tt)*] [$($_tv:tt)*] [$($tf:tt)*] [$($tb:tt)*]
        functional $_n3:literal [$($fe:tt)*] [$($fv:tt)*] [$($_ff:tt)*] [$($_fb:tt)*]
    ]) => {
        #[allow(clippy::all)]
        impl<$($g)*> $crate::paths::PathModel for $($ty)* where $($wh)* {
            type P<__RaadsT: Copy> = [__RaadsT; $np];
            type C<__RaadsT: Copy> = [__RaadsT; $nc];
            type S<__RaadsT: Copy> = [__RaadsT; $ns];
            type Z<__RaadsT: Copy> = [__RaadsT; $nz];
            #[doc(hidden)]
            const STEP_NRES: usize = $nres;
            $($c)*
            $($se)* $($sv)* $($ie)* $($iv)* $($te)* $($tf)* $($tb)* $($fe)* $($fv)*
        }
    };
    ([$($krate:tt)*] $($rest:tt)*) => {
        $($krate)*::__private::check_kernels! {
            $crate::__raads_paths_model;
            spec {
                fn setup(p: [N; NP], #[data] n_steps: N) -> [N; NC];
                fn init(p: [N; NP]) -> [N; NS];
                fn step(s: [N; NS], c: [N; NC], #[data] z: [N; NZ]) -> [N; NS];
                fn functional(s: [N; NS], p: [N; NP], c: [N; NC]) -> N;
            }
            $($rest)*
        }
    };
}

#[doc(hidden)]
pub use crate::__raads_paths_model as __raads_model;

/// Path means of the functional and its gradient, and their tangents, one per tangent
/// direction of the value type (none for a [`Lanes`] pack, one for [`Dual`](raads::Dual),
/// `K` for [`DualVec`](raads::DualVec)). For parameters seeded with directions `v_k`,
/// `value_tangent[k]` = grad · v_k and `grad_tangent[k]` = H v_k.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[must_use]
pub struct PathGradient<G, T, TG> {
    /// Mean of the functional.
    pub value: f64,
    /// Standard error of `value`: sample standard deviation over paths / sqrt(paths).
    pub std_err: f64,
    /// d value / d params.
    pub grad: G,
    /// d value / d params · v_k, per direction.
    pub value_tangent: T,
    /// H v_k, per direction.
    pub grad_tangent: TG,
}

/// What [`gradient`] returns for model `M` and value type `R`.
pub type PathGradientOf<M, R> = PathGradient<<M as PathModel>::P<f64>, <R as Real>::T<f64>, <R as Real>::T<<M as PathModel>::P<f64>>>;

/// Mean of the functional and its standard error, from [`value`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[must_use]
pub struct PathValue {
    /// Mean of the functional.
    pub value: f64,
    /// Standard error of `value`: sample standard deviation over paths / sqrt(paths).
    pub std_err: f64,
}

/// Terms (path blocks × steps) summed in `R` before the sums are flushed into `f64`. Bounds the
/// accumulation error by the block (not by `n_paths`), which matters for `f32`.
const FLUSH_TERMS: usize = 4096;

/// Checks the arguments; returns the normals one path block reads (`n_steps * NZ * W`).
fn path_stride(n_steps: usize, n_paths: usize, nz: usize, w: usize, n_normals: usize) -> usize {
    assert!(n_steps >= 1 && n_paths >= 1 && nz >= 1, "need n_steps, n_paths and NZ >= 1");
    assert!(n_paths.is_multiple_of(w), "n_paths = {n_paths} must be a multiple of the lane count W = {w}");
    assert!(n_normals >= n_paths * n_steps * nz, "need n_paths * n_steps * NZ normals");
    n_steps * nz * w
}

/// Element type of the normals: `f32` (half the memory traffic) or `f64`, converted to the pack.
pub trait Draw: Copy {
    /// First `W` draws of `src` as a pack.
    fn load<L: Lanes>(src: &[Self]) -> L;
}

impl Draw for f32 {
    #[inline(always)]
    fn load<L: Lanes>(src: &[f32]) -> L {
        L::load_f32(src)
    }
}

impl Draw for f64 {
    #[inline(always)]
    fn load<L: Lanes>(src: &[f64]) -> L {
        L::load_f64(src)
    }
}

/// The normals passed to [`gradient`] and [`value`]: a slice, `Vec` or array of [`Draw`]s.
pub trait Normals {
    /// `f32` or `f64`.
    type Draw: Draw;
    /// The draws as a slice.
    fn draws(&self) -> &[Self::Draw];
}

impl<D: Draw> Normals for [D] {
    type Draw = D;
    fn draws(&self) -> &[D] {
        self
    }
}

impl<D: Draw> Normals for Vec<D> {
    type Draw = D;
    fn draws(&self) -> &[D] {
        self
    }
}

impl<D: Draw, const N: usize> Normals for [D; N] {
    type Draw = D;
    fn draws(&self) -> &[D] {
        self
    }
}

/// `NZ` normals of one step, converted to `R`.
#[inline(always)]
fn load_z<M: PathModel, R: Real, D: Draw>(zk: &[D]) -> M::Z<R> {
    let w = R::Lanes::W;
    Arr::from_fn(|j| R::lift(D::load(&zk[j * w..])))
}

/// Mean functional of the first path block, as a pack. Sums are taken around it, so the sum
/// of squares does not cancel against the squared mean when the values' spread is small next
/// to their mean (`f32`).
fn shift<M: PathModel, R: Real, D: Draw>(params: &M::P<R>, c: &M::C<R>, s0: &M::S<R>, path: &[D]) -> R::Lanes {
    let mut s = *s0;
    for zk in path.chunks_exact(<M::Z<R> as Arr<R>>::LEN * R::Lanes::W) {
        s = M::step_eval(&s, c, &load_z::<M, R, D>(zk));
    }
    R::Lanes::from(M::functional_eval(&s, params, c).value().sum() / R::Lanes::W as f64)
}

/// Mean of the functional and its gradient (path means) in one pass.
///
/// `n_paths` counts scalar paths and must be a multiple of `W` (the lanes of `R`): lane `l`
/// of path block `i` is one path. `normals` holds `n_paths * n_steps * NZ` draws: block `i`
/// starts at `i * n_steps * NZ * W`, its step `k` at `+ k * NZ * W`, draw `j` at `+ j * W`.
///
/// Sums over paths are taken in `R` within groups of `ceil(4096 / n_steps)` path blocks, then
/// in `f64`. The reverse of `init` and `setup` is linear in its seeds, so it runs once per
/// group, on that group's sums.
///
/// # Panics
///
/// If `n_steps`, `n_paths` or `NZ` is zero, `n_paths` is not a multiple of `W`, or `normals`
/// is too short.
pub fn gradient<M: PathModel, R: Real>(
    params: &M::P<R>,
    n_steps: usize,
    n_paths: usize,
    normals: &(impl Normals + ?Sized),
) -> PathGradientOf<M, R> {
    let normals = normals.draws();
    let w = R::Lanes::W;
    let nz = <M::Z<R> as Arr<R>>::LEN;
    let nres = M::STEP_NRES;
    let stride = path_stride(n_steps, n_paths, nz, w, normals.len());

    // path-invariant parts: once per run
    let nd = R::from(n_steps as f64);
    let c = M::setup_eval(params, nd);
    let s0 = M::init_eval(params);
    let k = shift::<M, R, _>(params, &c, &s0, &normals[..stride]);
    let k_r = R::lift(k);

    let mut res = vec![R::from(0.0); n_steps * nres];
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    let mut value_tangent: R::T<f64> = Arr::splat(0.0);
    let mut grad: M::P<f64> = Arr::splat(0.0);
    let mut grad_tangent: R::T<M::P<f64>> = Arr::splat(Arr::splat(0.0));
    let flush = FLUSH_TERMS.div_ceil(n_steps);
    for block in normals[..n_paths * n_steps * nz].chunks(flush * stride) {
        let mut acc = R::neg_zero();
        let mut acc_sq = R::Lanes::neg_zero();
        let mut a_p: M::P<R> = Arr::splat(R::neg_zero());
        let mut a_c: M::C<R> = Arr::splat(R::neg_zero());
        let mut a_s0: M::S<R> = Arr::splat(R::neg_zero());
        for path in block.chunks_exact(stride) {
            let mut s = s0;
            for (k, zk) in path.chunks_exact(nz * w).enumerate() {
                let z = load_z::<M, R, _>(zk);
                s = if nres == 0 { M::step_eval(&s, &c, &z) } else { M::step_fwd(&s, &c, &z, &mut res[k * nres..(k + 1) * nres]) };
            }
            let mut a_s: M::S<R> = Arr::splat(R::neg_zero());
            let d = M::functional_vjp(&s, params, &c, R::from(1.0), &mut a_s, &mut a_p, &mut a_c) - k_r;
            acc += d;
            acc_sq = d.value().mul_add(d.value(), acc_sq);
            for (k, zk) in path.chunks_exact(nz * w).enumerate().rev() {
                let z = load_z::<M, R, _>(zk);
                let mut a_prev: M::S<R> = Arr::splat(R::neg_zero());
                M::step_bwd(&z, &res[k * nres..(k + 1) * nres], &a_s, &mut a_prev, &mut a_c);
                a_s = a_prev;
            }
            for (a0, a) in a_s0.as_mut().iter_mut().zip(a_s.as_ref()) {
                *a0 += *a;
            }
        }

        M::init_vjp(params, &a_s0, &mut a_p);
        M::setup_vjp(params, nd, &a_c, &mut a_p);
        sum += acc.value().sum();
        sum_sq += acc_sq.sum();
        for (vt, t) in value_tangent.as_mut().iter_mut().zip(acc.tangents().as_ref()) {
            *vt += t.sum();
        }
        for (j, a) in a_p.as_ref().iter().enumerate() {
            grad.as_mut()[j] += a.value().sum();
            for (gt, t) in grad_tangent.as_mut().iter_mut().zip(a.tangents().as_ref()) {
                gt.as_mut()[j] += t.sum();
            }
        }
    }

    let scale = 1.0 / n_paths as f64;
    let tangents = grad_tangent.as_mut().iter_mut().flat_map(|g| g.as_mut());
    for g in grad.as_mut().iter_mut().chain(value_tangent.as_mut()).chain(tangents) {
        *g *= scale;
    }
    let PathValue { value, std_err } = mean_and_err(k.lane(0), sum, sum_sq, n_paths);
    PathGradient { value, std_err, grad, value_tangent, grad_tangent }
}

/// Mean and `sqrt(sample variance / n)` of `n` values from the sums of `v - k` and `(v - k)²`.
fn mean_and_err(k: f64, sum: f64, sum_sq: f64, n: usize) -> PathValue {
    let nf = n as f64;
    let std_err = if n < 2 { f64::NAN } else { ((sum_sq - sum * sum / nf).max(0.0) / ((nf - 1.0) * nf)).sqrt() };
    PathValue { value: k + sum / nf, std_err }
}

/// Mean of the functional only, and its standard error: no residuals saved, no reverse
/// sweep. Same arguments, normals layout and panics as [`gradient`].
pub fn value<M: PathModel, R: Real>(params: &M::P<R>, n_steps: usize, n_paths: usize, normals: &(impl Normals + ?Sized)) -> PathValue {
    let normals = normals.draws();
    let w = R::Lanes::W;
    let nz = <M::Z<R> as Arr<R>>::LEN;
    let stride = path_stride(n_steps, n_paths, nz, w, normals.len());
    let c = M::setup_eval(params, R::from(n_steps as f64));
    let s0 = M::init_eval(params);
    let k = shift::<M, R, _>(params, &c, &s0, &normals[..stride]);
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    for block in normals[..n_paths * n_steps * nz].chunks(FLUSH_TERMS.div_ceil(n_steps) * stride) {
        let (mut acc, mut acc_sq) = (R::Lanes::neg_zero(), R::Lanes::neg_zero());
        for path in block.chunks_exact(stride) {
            let mut s = s0;
            for zk in path.chunks_exact(nz * w) {
                s = M::step_eval(&s, &c, &load_z::<M, R, _>(zk));
            }
            let d = M::functional_eval(&s, params, &c).value() - k;
            acc += d;
            acc_sq = d.mul_add(d, acc_sq);
        }
        sum += acc.sum();
        sum_sq += acc_sq.sum();
    }
    mean_and_err(k.lane(0), sum, sum_sq, n_paths)
}
