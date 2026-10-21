//! Path engine on models with closed forms, f64:
//! - Brownian motion with drift, value X_T² (no residuals) and geometric growth, value X_T
//!   (the step saves residuals): value and gradient vs the closed form within 4 standard errors;
//! - dot test: the value tangent with Dual parameters (forward mode through the engine) ==
//!   grad · v from the reverse sweep;
//! - gradient == central differences of the value on the same normals;
//! - Hessian in one pass (`DualVec`, unit seeds) == one `Dual` pass per column, and symmetric;
//! - `f64` normals holding the same values give the same result as `f32` normals;
//! - `value` == `gradient`'s value, and the argument checks.

use raads::lanes::F64x4;
use raads::{AdNum, Arr, Dual, DualVec, kernels};
use raads_engines::paths::{self, PathModel};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

type V = F64x4;
const N_STEPS: usize = 16;
const N_PATHS: usize = 200_000;

/// X_0 = x0, X_T = x0 + mu T + sigma W_T; value X_T². params: x0, mu, sigma, T.
pub struct Drift;

#[kernels(raads_engines::paths)]
impl Drift {
    pub fn setup<N: AdNum>(p: [N; 4], #[data] n_steps: N) -> [N; 2] {
        let dt = p[3] / n_steps;
        [p[1] * dt, p[2] * dt.sqrt()]
    }
    pub fn init<N: AdNum>(p: [N; 4]) -> [N; 1] {
        [p[0]]
    }
    pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] + c[1] * z[0]]
    }
    pub fn functional<N: AdNum>(s: [N; 1], _p: [N; 4], _c: [N; 2]) -> N {
        s[0] * s[0]
    }
}

/// E[X_T²] = (x0 + mu T)² + sigma² T.
fn drift_exact(p: [f64; 4]) -> f64 {
    let m = p[0] + p[1] * p[3];
    m * m + p[2] * p[2] * p[3]
}

/// X_{k+1} = X_k exp((mu - sigma²/2) dt + sigma √dt z); value X_T. params: x0, mu, sigma, T.
pub struct Growth;

#[kernels(raads_engines::paths)]
impl Growth {
    pub fn setup<N: AdNum>(p: [N; 4], #[data] n_steps: N) -> [N; 2] {
        let dt = p[3] / n_steps;
        [(p[1] - 0.5 * p[2] * p[2]) * dt, p[2] * dt.sqrt()]
    }
    pub fn init<N: AdNum>(p: [N; 4]) -> [N; 1] {
        [p[0]]
    }
    pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] * (c[0] + c[1] * z[0]).exp()]
    }
    pub fn functional<N: AdNum>(s: [N; 1], _p: [N; 4], _c: [N; 2]) -> N {
        s[0]
    }
}

/// E[X_T] = x0 exp(mu T).
fn growth_exact(p: [f64; 4]) -> f64 {
    p[0] * (p[1] * p[3]).exp()
}

fn normals(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| StandardNormal.sample(&mut rng)).collect()
}

fn lanes<M: PathModel>(p: [f64; 4]) -> M::P<V> {
    Arr::from_fn(|i| V::from(p[i]))
}

/// `grad_tol`: 4 standard errors of each gradient entry (RMS error over 10 seeds).
fn verify<M: PathModel>(label: &str, p: [f64; 4], exact: fn([f64; 4]) -> f64, grad_tol: [f64; 4]) {
    let z = normals(N_PATHS * N_STEPS, 42);
    let r = paths::gradient::<M, V>(&lanes::<M>(p), N_STEPS, N_PATHS, &z);
    println!("{label}: value {} ± {} grad {:?}", r.value, r.std_err, r.grad.as_ref());
    let e = exact(p);
    assert!((r.value - e).abs() < 4.0 * r.std_err, "{label}: value {} vs {e}", r.value);
    for j in 0..4 {
        let h = 1e-6 * p[j].abs().max(1e-2);
        let (mut a, mut b) = (p, p);
        a[j] += h;
        b[j] -= h;
        let want = (exact(a) - exact(b)) / (2.0 * h);
        assert!((r.grad.as_ref()[j] - want).abs() < grad_tol[j], "{label}: grad {j} {} vs closed form {want}", r.grad.as_ref()[j]);
    }

    // dot test
    let v = |j: usize| 0.3 + 0.1 * j as f64;
    let pd: M::P<Dual<V>> = Arr::from_fn(|j| Dual::new(V::from(p[j]), V::from(v(j))));
    let d = paths::gradient::<M, Dual<V>>(&pd, N_STEPS, N_PATHS, &z);
    let gv: f64 = (0..4).map(|j| r.grad.as_ref()[j] * v(j)).sum();
    assert!((d.value_tangent[0] - gv).abs() <= 1e-12 * gv.abs(), "{label}: dot test {} vs {gv}", d.value_tangent[0]);

    // Hessian in one pass == one Dual pass per column, and symmetric
    let hv = paths::gradient::<M, DualVec<V, 4>>(&Arr::from_fn(|j| DualVec::unit(V::from(p[j]), j)), N_STEPS, N_PATHS, &z);
    assert_eq!(hv.grad.as_ref(), r.grad.as_ref(), "{label}: DualVec gradient");
    for k in 0..4 {
        let e: M::P<Dual<V>> = Arr::from_fn(|j| Dual::new(V::from(p[j]), V::from(if j == k { 1.0 } else { 0.0 })));
        let hd = paths::gradient::<M, Dual<V>>(&e, N_STEPS, N_PATHS, &z);
        for i in 0..4 {
            let (a, b) = (hv.grad_tangent[k].as_ref()[i], hd.grad_tangent[0].as_ref()[i]);
            // tangents round differently (shared reciprocals), compounded over the steps
            assert!((a - b).abs() <= 1e-10 * b.abs().max(1.0), "{label}: H[{i}][{k}] {a} vs Dual {b}");
            let t = hv.grad_tangent[i].as_ref()[k];
            assert!((a - t).abs() <= 1e-9 * a.abs().max(1.0), "{label}: H[{i}][{k}] {a} vs H[{k}][{i}] {t}");
        }
    }

    // central differences on the same normals
    for j in 0..4 {
        let h = 1e-5 * p[j].abs().max(1e-2);
        let (mut a, mut b) = (p, p);
        a[j] += h;
        b[j] -= h;
        let value = |q| paths::value::<M, V>(&lanes::<M>(q), N_STEPS, N_PATHS, &z).value;
        let fd = (value(a) - value(b)) / (2.0 * h);
        let g = r.grad.as_ref()[j];
        assert!((g - fd).abs() <= 1e-6 * fd.abs().max(1e-2), "{label}: grad {j} {g} vs central differences {fd}");
    }

    let z64: Vec<f64> = z.iter().map(|&v| v as f64).collect();
    let r64 = paths::gradient::<M, V>(&lanes::<M>(p), N_STEPS, N_PATHS, &z64);
    assert_eq!((r64.value, r64.grad.as_ref()), (r.value, r.grad.as_ref()), "{label}: f64 normals");

    let pv = paths::value::<M, V>(&lanes::<M>(p), N_STEPS, N_PATHS, &z);
    assert!((pv.value - r.value).abs() <= 1e-12 * r.value.abs(), "{label}: value {} vs gradient's {}", pv.value, r.value);
    assert!((pv.std_err - r.std_err).abs() <= 1e-9 * r.std_err, "{label}: std_err {} vs gradient's {}", pv.std_err, r.std_err);
}

#[test]
fn drift() {
    const { assert!(<Drift as PathModel>::STEP_NRES == 0) };
    verify::<Drift>("drift", [1.0, 0.5, 0.8, 2.0], drift_exact, [0.014, 0.028, 0.071, 0.021]);
}

#[test]
fn growth() {
    const { assert!(<Growth as PathModel>::STEP_NRES > 0) };
    verify::<Growth>("growth", [1.0, 0.05, 0.3, 1.5], growth_exact, [0.0031, 0.0046, 0.0123, 0.0014]);
}

#[test]
#[should_panic(expected = "need n_steps, n_paths and NZ >= 1")]
fn zero_steps() {
    let _ = paths::gradient::<Drift, V>(&lanes::<Drift>([1.0; 4]), 0, 4, &[0f32; 0]);
}

#[test]
#[should_panic(expected = "n_paths = 6 must be a multiple of the lane count W = 4")]
fn partial_block() {
    let _ = paths::value::<Drift, V>(&lanes::<Drift>([1.0; 4]), 2, 6, &[0.0; 16]);
}

#[test]
#[should_panic(expected = "need n_paths * n_steps * NZ normals")]
fn short_normals() {
    let _ = paths::gradient::<Drift, V>(&lanes::<Drift>([1.0; 4]), 2, 8, &[0.0; 15]);
}
