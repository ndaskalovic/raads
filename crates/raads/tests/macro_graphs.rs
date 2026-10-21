//! `#[kernel]` front end on tricky sources: the graph the macro builds from tokens equals
//! the graph the tracer records by running the same function, and the compiled code
//! computes what the eager function computes.

use raads::__private::{Tracer, trace};
use raads::lanes::F64x4;
use raads::{AdNum, Lanes, kernel};

const K: f64 = 1.5;

/// Negative `f64` receivers and integer literal arguments.
#[kernel]
fn negatives<N: AdNum>(x: [N; 2]) -> N {
    let k = -1.0_f64;
    x[0] * (-1.0_f64).exp() + x[1] * k.exp() + x[0] * (-4.0_f64).abs() + x[1] * 2.0_f64.powi(2) - 3.0 + x[0] * -2.0 - K * x[1]
}

#[kernel]
fn rebinding<N: AdNum>(x: [N; 2]) -> N {
    let mut u = x[0] * x[1];
    u = u.exp() + x[0];
    let u = u * 2.0;
    u - x[1]
}

#[kernel]
fn patterns<N: AdNum>(x: [N; 3]) -> [N; 2] {
    let (a, b) = (x[0], x[1] - x[2]);
    let [c, _, e] = x;
    let (p, q): (N, N) = (a * c, b / e);
    [p + q, c.maximum(e) - a.minimum(b)]
}

#[kernel]
fn constants<N: AdNum>(x: [N; 2], #[data] d: N) -> [N; 4] {
    let s = N::select_gt(d, N::from(0.5), x[0].smooth_relu(0.1), -x[1]);
    [K - x[0], 2.0 / x[1], x[0] / K, s + std::f64::consts::PI * d]
}

/// Integer literals as numbers, and plain `f64` arguments where an `N` is expected: the
/// rewritten function makes them `f64` and `N::from(k)` constants, in the tracer's order.
#[kernel]
fn plain_numbers<N: AdNum>(x: [N; 2], #[data] d: N) -> [N; 5] {
    let k = 2;
    let s = N::select_gt(x[0] * 3, K, (x[1] - 1).exp(), 0);
    [x[0] * 2 + 1, 4 / x[1] - x[0].powf(2), N::from(3) * k - x[1].maximum(K) + x[0].minimum(0.5), s, d.maximum(1) * x[0]]
}

/// Data and differentiable parameters interleaved: inputs are numbered over the
/// differentiable groups, data over the data groups.
#[kernel]
fn ordering<N: AdNum>(#[data] z: N, x: N, #[data] w: [N; 2], y2: [N; 2]) -> N {
    (x * z + y2[0] * w[1] - y2[1] / w[0]).exp()
}

/// Stop-gradient: the detached subgraph is evaluated but needs no adjoints or residuals.
#[kernel]
fn stopped<N: AdNum>(x: [N; 2]) -> N {
    (x[0] * x[1]).exp().sqrt().detach() * x[0]
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-14 * (1.0 + b.abs())
}

#[test]
fn negative_receivers_and_integer_arguments() {
    let x = [0.3, 0.7];
    let want = x[0] * (-1.0f64).exp() + x[1] * (-1.0f64).exp() + 4.0 * x[0] + 4.0 * x[1] - 3.0 - 2.0 * x[0] - K * x[1];
    assert!(close(negatives(x), want));
    assert!(close(negatives::eval(&x), want));
    assert!(close(negatives::eval(&x.map(F64x4::from)).lane(0), want));
    assert_eq!(negatives::graph(), trace(|x, _: [Tracer; 0]| [negatives(x)]));
}

#[test]
fn rebinding_matches() {
    let x = [0.3, 0.7];
    assert!(close(rebinding::eval(&x), rebinding(x)));
    assert_eq!(rebinding::graph(), trace(|x, _: [Tracer; 0]| [rebinding(x)]));
}

#[test]
fn patterns_match() {
    let x = [0.3, 0.7, 1.1];
    let (y, yc) = (patterns(x), patterns::eval(&x));
    assert!((0..2).all(|k| close(yc[k], y[k])));
    assert_eq!(patterns::graph(), trace(|x, _: [Tracer; 0]| patterns(x)));
}

#[test]
fn constants_match() {
    for d in [0.2, 0.9] {
        let x = [0.3, 0.7];
        let (y, yc) = (constants(x, d), constants::eval(&x, d));
        assert!((0..4).all(|k| close(yc[k], y[k])));
    }
    assert_eq!(constants::graph(), trace(|x, d: [Tracer; 1]| constants(x, d[0])));
}

#[test]
fn plain_numbers_match() {
    for (x, d) in [([0.3, 0.7], 0.2f64), ([0.6, 1.9], 1.4)] {
        let x: [f64; 2] = x;
        let want = [
            2.0 * x[0] + 1.0,
            4.0 / x[1] - x[0] * x[0],
            6.0 - x[1].max(K) + x[0].min(0.5),
            if 3.0 * x[0] > K { (x[1] - 1.0).exp() } else { 0.0 },
            d.max(1.0) * x[0],
        ];
        let (y, yc) = (plain_numbers(x, d), plain_numbers::eval(&x, d));
        assert!((0..5).all(|k| close(y[k], want[k]) && close(yc[k], want[k])), "{y:?} {yc:?} {want:?}");
    }
    assert_eq!(plain_numbers::graph(), trace(|x, d: [Tracer; 1]| plain_numbers(x, d[0])));
}

#[test]
fn data_ordering() {
    let (z, x, w, y2) = (0.4, 0.3, [1.2, 0.8], [0.5, 0.9]);
    assert!(close(ordering::eval(z, x, &w, &y2), ordering(z, x, w, y2)));
    let g = trace(|x: [Tracer; 3], d: [Tracer; 3]| [ordering(d[0], x[0], [d[1], d[2]], [x[1], x[2]])]);
    assert_eq!(ordering::graph(), g);
}

#[test]
fn stop_gradient_saves_only_its_value() {
    assert_eq!(stopped::graph(), trace(|x, _: [Tracer; 0]| [stopped(x)]));
    assert_eq!(stopped::NRES, 1);
    let x = [0.3, 0.7];
    let mut a = [0.0; 2];
    let y = stopped::vjp(&x, 1.0, &mut a);
    let s = (x[0] * x[1]).exp().sqrt();
    assert!(close(y, s * x[0]));
    assert_eq!(a, [s, 0.0]);

    // fwd / bwd agree with vjp
    let mut res = [0.0; stopped::NRES];
    assert!(close(stopped::fwd(&x, &mut res), y));
    let mut b = [0.0; 2];
    stopped::bwd(&res, 1.0, &mut b);
    assert_eq!(a, b);
}
