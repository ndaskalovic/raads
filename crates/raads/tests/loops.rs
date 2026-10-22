//! Kernels with loops, f64:
//! 1. the macro's graph, unrolled, == the tracer's graph of the eager function
//! 2. primal == the eager function
//! 3. reverse gradient == forward-mode tangents (Dual) of the eager function
//! 4. `fwd` then `bwd` == `vjp`; SIMD lanes == scalar
//! 5. forward-over-reverse Hessian-vector product == central differences of the gradient

use raads::__private::codegen::unroll;
use raads::__private::{Tracer, trace};
use raads::lanes::F64x4;
use raads::{AdNum, Dual, Lanes, kernel};

/// A carried sum and input elements read with a loop index.
#[kernel]
fn chain<N: AdNum>(x: [N; 6], #[data] w: [N; 6]) -> [N; 1] {
    let mut s = N::from(0.0);
    let mut c = x[0] * 0.1;
    for i in 0..5 {
        c = c + x[i] * w[i];
        s = s + x[i] * x[i + 1] * (-c).exp();
    }
    [s * c]
}

/// LIBOR-like: a local array updated in a triangular loop nest with an inner carried value,
/// and a top-level value used two loops deep.
#[kernel]
fn triangle<N: AdNum>(p: [N; 8], #[data] z: [N; 3]) -> [N; 2] {
    let d = p[0];
    let mut l = [N::from(0.0); 4];
    for i in 0..4 {
        l[i] = p[1 + i];
    }
    for n in 0..3 {
        let sq = d.sqrt() * z[n];
        let mut v = N::from(0.0);
        for i in n + 1..4 {
            let lam = p[5 + i - n - 1];
            let c = d * lam;
            v = v + c * l[i] / (d * l[i] + 1.0);
            l[i] = l[i] * (c * v + lam * (sq - c * 0.5)).exp();
        }
    }
    let mut b = N::from(1.0);
    for k in 0..4 {
        b = b / (d * l[k] + 1.0);
    }
    [b, (l[3] - 0.04).relu() + l[2].maximum(l[1])]
}

/// An array filled with an input and returned, `select_gt` in a loop, the loop index as a
/// number, an inner loop that runs 0, 0 and 1 times, constant indices into a local array.
#[kernel]
fn misc<N: AdNum>(x: [N; 3], #[data] d: [N; 1]) -> [N; 3] {
    let mut a = [x[0]; 3];
    let mut t = N::from(1.0);
    for k in 0..3 {
        a[k] = N::select_gt(x[1], d[0], a[k] * x[2], a[k] + (k as f64) * x[1]);
        t = t * a[k];
        for _j in 1..k {
            t = t + 1.0
        }
    }
    a[1] = t * a[2];
    a
}

const LAST: usize = 4;
const IDX: [usize; 4] = [2, 0, 3, 1];
const W: [f64; 4] = [0.5, -1.0, 2.0, 0.25];

fn pick(h: usize) -> usize {
    h / 2
}

fn weight(h: usize) -> f64 {
    h as f64 * 0.1 + 0.3
}

fn damp(w: f64) -> f64 {
    0.1 / (1.0 + w)
}

/// Plain Rust indices and numbers: const arrays, functions, integer locals (one used in an
/// inner loop), a plain f64 local passed to a function, and computed indices at the top level
/// (an input and a data element).
#[kernel]
fn plain<N: AdNum>(x: [N; 5], #[data] d: [N; 4]) -> [N; 2] {
    let last: usize = LAST;
    let mut a = [x[0]; 4];
    let mut s = x[last] * d[IDX[0]];
    for i in 0..4 {
        let k: usize = IDX[i];
        let w = weight(i + 1);
        a[k] = a[k] * x[pick(i + 2)] + w * W[i] * d[k];
        s = s + (1.0 - w) * a[IDX[3 - i]] + (k as f64) * x[k + 1];
        for j in 0..i {
            s = s * (W[j] * weight(k + j) * damp(w)).exp() + x[pick(j + k)] * d[(j + k) % 4];
        }
    }
    [s, a[pick(LAST - 1)]]
}

/// An inner loop variable that shadows an outer one: plain locals keep the outer value.
#[kernel]
fn shadowed<N: AdNum>(x: [N; 4], #[data] d: [N; 1]) -> [N; 1] {
    let mut s = N::from(0.0);
    for i in 0..2 {
        let k: usize = i;
        let w = i as f64;
        for i in 0..2 {
            s = s + w * x[i] + x[k + 2] * (i as f64 + 1.0);
        }
    }
    [s * d[0]]
}

/// A chain of plain locals, each read several times: each is computed once, so the
/// expansion grows linearly with the chain.
#[kernel]
fn locals<N: AdNum>(x: [N; 2], #[data] d: [N; 1]) -> [N; 1] {
    let mut s = x[0];
    for i in 0..3 {
        let a0 = i as f64 * 0.25 + 1.0;
        let a1 = a0 * a0 - a0 * 0.5;
        let a2 = a1 * a1 - a1 * 0.5;
        let a3 = a2 * a2 - a2 * 0.5;
        let a4 = a3 * a3 - a3 * 0.5;
        let a5 = a4 * a4 - a4 * 0.5;
        let a6 = a5 * a5 - a5 * 0.5;
        let a7 = a6 * a6 - a6 * 0.5;
        let a8 = a7 * a7 - a7 * 0.5;
        let a9 = a8 * a8 - a8 * 0.5;
        let a10 = a9 * a9 - a9 * 0.5;
        let a11 = a10 * a10 - a10 * 0.5;
        let a12 = a11 * a11 - a11 * 0.5;
        let a13 = a12 * a12 - a12 * 0.5;
        let a14 = a13 * a13 - a13 * 0.5;
        let a15 = a14 * a14 - a14 * 0.5;
        let a16 = a15 * a15 - a15 * 0.5;
        let k0: usize = i % 2;
        let k1: usize = k0 * k0 + k0 - k0;
        s = s * (a16 + 1.0) + x[k1] * d[0] * a8;
    }
    [s]
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// Checks 1-5 for kernel `$k` with `NX` inputs `$x`, data `$d` and `NO` outputs.
macro_rules! check {
    ($k:ident, $x:expr, $d:expr, $no:literal) => {{
        let (x, d) = ($x, $d);
        const NX: usize = $x.len();
        assert_eq!(unroll(&$k::graph()), trace(|x: [Tracer; NX], d| $k(x, d)), "graph");
        let want = $k(x, d);
        let y = $k::eval(&x, &d);
        for o in 0..$no {
            assert!(close(y[o], want[o], 1e-13), "primal {o}: {} vs {}", y[o], want[o]);
        }
        for o in 0..$no {
            let a_y: [f64; $no] = std::array::from_fn(|k| (k == o) as u8 as f64);
            let mut g = [0.0; NX];
            assert_eq!($k::vjp(&x, &d, &a_y, &mut g), y);
            for j in 0..NX {
                let xd: [Dual<f64>; NX] = std::array::from_fn(|k| Dual::new(x[k], (k == j) as u8 as f64));
                let t = $k(xd, d.map(Dual::from))[o].t;
                assert!(close(g[j], t, 1e-12), "d y{o} / d x{j}: reverse {} vs forward {t}", g[j]);
            }
            let mut res = vec![0.0; $k::NRES];
            assert_eq!($k::fwd(&x, &d, &mut res), y);
            let mut g2 = [0.0; NX];
            $k::bwd(&d, &res, &a_y, &mut g2);
            assert_eq!(g2, g, "fwd + bwd == vjp");

            let xl: [F64x4; NX] = std::array::from_fn(|k| F64x4::from_fn(|l| x[k] * (1.0 + 0.01 * l as f64)));
            let mut gl = [F64x4::from(0.0); NX];
            $k::vjp(&xl, &d.map(F64x4::from), &a_y.map(F64x4::from), &mut gl);
            for l in 0..4 {
                let mut gs = [0.0; NX];
                $k::vjp(&xl.map(|v| v.lane(l)), &d, &a_y, &mut gs);
                for j in 0..NX {
                    assert!(close(gl[j].lane(l), gs[j], 1e-14), "lane {l}");
                }
            }

            let v: [f64; NX] = std::array::from_fn(|k| 0.3 + 0.1 * k as f64);
            let xv: [Dual<f64>; NX] = std::array::from_fn(|k| Dual::new(x[k], v[k]));
            let mut h = [Dual::from(0.0); NX];
            $k::vjp(&xv, &d.map(Dual::from), &a_y.map(Dual::from), &mut h);
            let eps = 1e-5;
            let grad_at = |s: f64| {
                let mut g = [0.0; NX];
                $k::vjp(&std::array::from_fn(|k| x[k] + s * v[k]), &d, &a_y, &mut g);
                g
            };
            let (gp, gm) = (grad_at(eps), grad_at(-eps));
            for j in 0..NX {
                let fd = (gp[j] - gm[j]) / (2.0 * eps);
                assert!(close(h[j].t, fd, 1e-6), "H v [{j}]: {} vs {fd}", h[j].t);
            }
        }
    }};
}

#[test]
fn carried_sum() {
    check!(chain, [0.3, 0.7, -0.2, 0.5, 1.1, 0.4], [0.5, -0.3, 0.8, 0.2, -0.6, 0.9], 1);
}

#[test]
fn triangular_nest() {
    check!(triangle, [0.05, 0.05, 0.06, 0.04, 0.055, 0.2, 0.25, 0.18], [0.3, -1.2, 0.7], 2);
}

#[test]
fn plain_rust() {
    check!(plain, [0.4, 0.9, 1.3, -0.2, 0.7], [0.5, -0.8, 1.1, 0.3], 2);
}

#[test]
fn shadowed_loop_variable() {
    check!(shadowed, [1.0, 2.0, 3.0, 5.0], [0.7], 1);
}

#[test]
fn chained_locals() {
    check!(locals, [0.6, -0.4], [0.3], 1);
}

#[test]
fn arrays_and_selects() {
    check!(misc, [0.4, 0.9, 1.3], [0.5], 3);
    check!(misc, [0.4, 0.2, 1.3], [0.5], 3);
}
