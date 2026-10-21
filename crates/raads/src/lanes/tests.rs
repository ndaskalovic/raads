//! Every pack op lane-by-lane against plain scalar arithmetic.

use super::*;
use crate::{AdNum, Dual};

fn check<L: Lanes>(name: &str) {
    let w = L::W;
    let xs: Vec<f64> = (0..w).map(|i| -3.0 + 6.5 * (i as f64 + 0.37) / w as f64).collect();
    let ys: Vec<f64> = (0..w).map(|i| 0.4 + 2.1 * ((i * 7 % w) as f64 + 0.5) / w as f64).collect();
    let x = L::from_fn(|i| xs[i]);
    let y = L::from_fn(|i| ys[i]);
    let f32_ = std::mem::size_of::<L::Elem>() == 4;
    let round = |v: f64| if f32_ { v as f32 as f64 } else { v };
    let lane = |v: L, i: usize| v.lane(i);
    for i in 0..w {
        let (a, b) = (round(xs[i]), round(ys[i]));
        let exact = |got: f64, want: f64, op: &str| {
            assert_eq!(got, round(want), "{name} {op} lane {i}: {got} vs {want}");
        };
        exact(lane(x + y, i), a + b, "add");
        exact(lane(x - y, i), a - b, "sub");
        exact(lane(x * y, i), a * b, "mul");
        exact(lane(x / y, i), a / b, "div");
        exact(lane(-x, i), -a, "neg");
        exact(lane(x.relu(), i), a.max(0.0), "relu");
        exact(lane(y.sqrt(), i), b.sqrt(), "sqrt");
        exact(lane(L::select_gt(x, y, x, y), i), if a > b { a } else { b }, "select_gt");
        // rounded once where the target has FMA (see `mul_add!` and `simd::fma`)
        let fused = cfg!(any(target_arch = "aarch64", target_feature = "fma"));
        let fma = |a: f64, b: f64, c: f64| match (f32_, fused) {
            (true, true) => (a as f32).mul_add(b as f32, c as f32) as f64,
            (true, false) => (a as f32 * b as f32 + c as f32) as f64,
            (false, true) => a.mul_add(b, c),
            (false, false) => a * b + c,
        };
        exact(lane(x.mul_add(y, x), i), fma(a, b, a), "mul_add");
        exact(lane(x.nmul_add(y, x), i), fma(-a, b, a), "nmul_add");
        let tol = if f32_ { 3e-7 } else { 1e-15 };
        let rel = |got: f64, want: f64, tol: f64, op: &str| {
            assert!(((got - want) / want).abs() <= tol, "{name} {op} lane {i}: {got} vs {want}");
        };
        rel(lane(x.exp(), i), a.exp(), tol, "exp");
        rel(lane((x * 25.0).exp(), i), round(a * 25.0).exp(), tol, "exp wide range");
        rel(lane(y.ln(), i), b.ln(), tol, "ln");
        // exp(k·ln x): the error of ln, scaled by |k ln x| ≤ 1.6 here, on top of exp's
        rel(lane(y.powf(1.7), i), b.powf(1.7), 2.0 * tol, "powf");
        for k in [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 0.5, -0.5] {
            rel(lane(y.powf(k), i), b.powf(k), tol, "powf small k");
            let (got, want) = (lane(x.powf(k), i), a.powf(k));
            if want.is_nan() {
                assert!(got.is_nan(), "{name} powf({a}, {k}) lane {i}: {got} vs NaN");
            } else {
                rel(got, want, tol, "powf small k, any sign");
            }
        }
    }
    let s: f64 = (0..w).map(|i| round(xs[i])).sum();
    assert!((x.sum() - s).abs() < 1e-12, "{name} sum");
    let mut buf = vec![L::Elem::default(); w];
    x.store(&mut buf);
    assert_eq!(L::load(&buf), x, "{name} load/store");
    let f32s: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
    for i in 0..w {
        assert_eq!(L::load_f32(&f32s).lane(i), round(f32s[i] as f64), "{name} load_f32");
        assert_eq!(L::load_f64(&xs).lane(i), round(xs[i]), "{name} load_f64");
    }
}

/// `exp` at edge inputs (overflow and underflow thresholds, ±inf, NaN, -0.0) in every lane,
/// against `std` in the pack's precision; `relu` of NaN and ±inf.
fn edges<L: Lanes>(name: &str) {
    let f32_ = std::mem::size_of::<L::Elem>() == 4;
    let (max, min_pos, tiny, tol) =
        if f32_ { (f32::MAX as f64, f32::MIN_POSITIVE as f64, 1.5e-45, 3e-7) } else { (f64::MAX, f64::MIN_POSITIVE, 5e-324, 1e-15) };
    let reference = |x: f64| if f32_ { (x as f32).exp() as f64 } else { x.exp() };
    let xs = [
        0.0,
        -0.0,
        1e-30,
        -1e-30,
        0.346_573_59,
        -0.346_573_59,
        10.0,
        -10.0,
        80.0,
        -80.0,
        86.9,
        -86.9,
        87.0,
        -87.0,
        -87.3,
        -87.4,
        -87.5,
        -87.6,
        -88.0,
        88.0,
        88.5,
        88.7,
        88.72,
        88.73,
        89.0,
        89.5,
        -100.0,
        -103.0,
        -104.0,
        -110.0,
        -200.0,
        100.0,
        700.0,
        708.0,
        709.0,
        709.5,
        709.7,
        709.78,
        709.79,
        710.0,
        711.0,
        -708.0,
        -708.3,
        -708.5,
        -708.7,
        -709.0,
        -720.0,
        -745.0,
        -746.0,
        -750.0,
        1e10,
        -1e10,
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    for (b, xb) in xs.chunks(L::W).enumerate() {
        let e = L::from_fn(|i| xb[i % xb.len()]).exp();
        for (i, &x) in xb.iter().enumerate() {
            let (got, want) = (e.lane(i), reference(x));
            let ok = if want.is_nan() {
                got.is_nan()
            } else if want.is_infinite() || got.is_infinite() {
                got == want || got.min(want) >= max * (1.0 - tol)
            } else if want < 1.5 * min_pos {
                // flushed to 0 by the exponent-bit backends, else (gradual) underflow
                got == 0.0 || (got - want).abs() <= tol * want + tiny
            } else {
                ((got - want) / want).abs() <= tol
            };
            assert!(ok, "{name} exp({x}) lane {}: {got:e} vs {want:e}", b * L::W + i);
            if x == f64::NEG_INFINITY || x < -760.0 {
                assert_eq!(got, 0.0, "{name} exp({x})");
            }
        }
    }
    let r = L::from_fn(|i| [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -2.0][i % 4]).relu();
    for i in 0..L::W {
        let got = r.lane(i);
        match i % 4 {
            0 => assert!(got.is_nan(), "{name} relu(NaN) = {got}"),
            1 => assert_eq!(got, f64::INFINITY, "{name} relu(inf)"),
            _ => assert_eq!(got, 0.0, "{name} relu(negative)"),
        }
    }
    // ln: zeros, negatives, ±inf, NaN, subnormals, extremes
    let reference_ln = |x: f64| if f32_ { (x as f32).ln() as f64 } else { x.ln() };
    let (max_e, sub) = if f32_ { (f32::MAX as f64, 1.0e-40) } else { (f64::MAX, 1.0e-310) };
    let xs = [0.0, -0.0, -1.0, -tiny, f64::NEG_INFINITY, f64::INFINITY, f64::NAN, tiny, sub, min_pos, max_e, 1.0, 1.0 + 1e-6, 0.5];
    for (b, xb) in xs.chunks(L::W).enumerate() {
        let l = L::from_fn(|i| xb[i % xb.len()]).ln();
        for (i, &x) in xb.iter().enumerate() {
            let (got, want) = (l.lane(i), reference_ln(x));
            let ok = if want.is_nan() {
                got.is_nan()
            } else if want.is_infinite() || want == 0.0 {
                got == want
            } else {
                ((got - want) / want).abs() <= tol
            };
            assert!(ok, "{name} ln({x}) lane {}: {got:e} vs {want:e}", b * L::W + i);
        }
    }
    // powf: k = 0, integer k on negative bases, zeros, non-integer k on negatives
    let reference_pow = |x: f64, k: f64| if f32_ { (x as f32).powf(k as f32) as f64 } else { x.powf(k) };
    let cases = [
        (0.0, 0.0),
        (f64::NAN, 0.0),
        (-2.0, 2.0),
        (-2.0, 3.0),
        (-2.0, -3.0),
        (-2.0, 4.0),
        (-0.0, 5.0),
        (-0.0, -5.0),
        (f64::NEG_INFINITY, 5.0),
        (-2.0, 0.5),
        (0.0, 2.5),
        (0.0, -1.0),
        (-0.0, -1.0),
        (-0.0, 3.0),
        (f64::INFINITY, -0.5),
        (f64::NAN, 2.0),
        (3.0, 1.0),
    ];
    for &(x, k) in &cases {
        let (got, want) = (L::from(x).powf(k).lane(0), reference_pow(x, k));
        let ok = if want.is_nan() {
            got.is_nan()
        } else if want.is_infinite() || want == 0.0 {
            got == want && got.is_sign_negative() == want.is_sign_negative()
        } else {
            ((got - want) / want).abs() <= 2.0 * tol
        };
        assert!(ok, "{name} powf({x}, {k}): {got:e} vs {want:e}");
    }
    // x^±0.5 follows sqrt at -0 and -inf (documented on powf_small), unlike std
    let (h, mh) = (|x: f64| L::from(x).powf(0.5).lane(0), |x: f64| L::from(x).powf(-0.5).lane(0));
    assert!(h(-0.0) == 0.0 && h(-0.0).is_sign_negative(), "{name} (-0)^0.5 = {}", h(-0.0));
    assert_eq!(mh(-0.0), f64::NEG_INFINITY, "{name} (-0)^-0.5");
    assert!(h(f64::NEG_INFINITY).is_nan() && mh(f64::NEG_INFINITY).is_nan(), "{name} (-inf)^±0.5");
    let d = Dual::new(L::from(f64::NAN), L::from(1.0)).relu();
    assert!(d.v.lane(0).is_nan(), "{name} Dual relu(NaN)");
}

fn check_all<L: Lanes>(name: &str) {
    check::<L>(name);
    edges::<L>(name);
}

#[test]
fn scalar_f64() {
    check_all::<f64>("f64");
}

/// Odd and single-lane widths, which no alias uses.
#[test]
fn other_widths() {
    use super::simd::{F32, F64};
    check_all::<F32<1>>("F32<1>");
    check_all::<F32<2>>("F32<2>");
    check_all::<F64<1>>("F64<1>");
    check_all::<F64<3>>("F64<3>");
}

/// Every width alias.
#[test]
fn aliases() {
    check_all::<F32x4>("F32x4");
    check_all::<F32x8>("F32x8");
    check_all::<F32x16>("F32x16");
    check_all::<F32x32>("F32x32");
    check_all::<F64x2>("F64x2");
    check_all::<F64x4>("F64x4");
    check_all::<F64x8>("F64x8");
    check_all::<F64x16>("F64x16");
}

/// Max relative error of `ln` over `n` inputs spread over every binade (subnormals included), plus
/// the neighbourhood of 1, against `std` in the pack's precision, in units of the type's epsilon.
fn ln_sweep<L: Lanes>(lo_log2: f64, hi_log2: f64, eps: f64) -> f64 {
    let f32_ = std::mem::size_of::<L::Elem>() == 4;
    let reference = |x: f64| if f32_ { (x as f32).ln() as f64 } else { x.ln() };
    let n = 1 << 20;
    let mut xs: Vec<f64> = (0..n).map(|j| ((j as f64 + 0.5) / n as f64 * (hi_log2 - lo_log2) + lo_log2).exp2()).collect();
    xs.extend((0..n).map(|j| 0.5 + 1.5 * (j as f64 + 0.5) / n as f64));
    xs.extend((1..60).flat_map(|k| [1.0 + (-k as f64).exp2(), 1.0 - (-k as f64).exp2()]));
    let mut worst = 0f64;
    for c in xs.chunks(L::W) {
        let l = L::from_fn(|i| c[i % c.len()]).ln();
        for (i, &x) in c.iter().enumerate() {
            let want = reference(x);
            if want != 0.0 {
                worst = worst.max(((l.lane(i) - want) / want).abs() / eps);
            }
        }
    }
    worst
}

#[test]
fn ln_accuracy() {
    let e32 = ln_sweep::<F32x8>(-149.0, 128.0, f32::EPSILON as f64);
    let e64 = ln_sweep::<F64x4>(-1074.0, 1024.0, f64::EPSILON);
    println!("ln max relative error: f32 {e32:.3} eps, f64 {e64:.3} eps");
    // the reference (the platform's libm) is itself up to an ulp off, so allow two
    assert!(e32 <= 2.0 && e64 <= 2.0, "ln: f32 {e32} eps, f64 {e64} eps");
}
