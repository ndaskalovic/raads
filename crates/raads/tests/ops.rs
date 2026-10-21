//! Every supported operation, compiled forward and reverse, f64, random inputs:
//! 1. primal == a plain f64 reference of the same formulas
//! 2. dot-product test: <w, J v> (forward mode, Dual) == <v, Jᵀ w> (reverse mode, vjp)
//! 3. reverse gradient == central differences of the primal
//! 4. forward-over-reverse Hessian == central differences of the gradient, and symmetric
//! 5. stop-gradient: detach contributes its value, not its derivative
//! 6. the macro's graph == the tracer's graph, with fixed node and residual counts

use raads::__private::{Tracer, trace};
use raads::lanes::F64x4;
use raads::{AdNum, Dual, DualVec, Lanes, Real, kernel};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type V = F64x4;
type D = Dual<V>;
const W: usize = 4;

/// One kernel exercising every operation, with shared subexpressions and an unused branch.
#[kernel]
fn all_ops<N: AdNum>(x: [N; 4], #[data] d: [N; 1]) -> [N; 3] {
    let a = x[0];
    let b = x[1];
    let c = x[2];
    let e = x[3];
    let z = d[0];
    let s = (a * b + c / e - a / 3.0 + 2.0 * b - (1.5 - c)) * z;
    let t = a.exp() + b.ln() + c.sqrt() + e.powf(1.7) + (-a) + 1.0 / b + (a - 0.3).relu();
    let u = a.maximum(b) * c.minimum(e) + N::select_gt(z, N::from(0.0), a * c, b + e);
    let _dead = (a * b).exp();
    [s * t, u / (t + 3.0) + s, (s + u).relu() + 0.25 * t + a.detach() * b * c.detach()]
}

fn reference(x: [f64; 4], z: f64) -> [f64; 3] {
    let [a, b, c, e] = x;
    let s = (a * b + c / e - a / 3.0 + 2.0 * b - (1.5 - c)) * z;
    let t = a.exp() + b.ln() + c.sqrt() + e.powf(1.7) - a + 1.0 / b + (a - 0.3).max(0.0);
    let u = a.max(b) * c.min(e) + if z > 0.0 { a * c } else { b + e };
    [s * t, u / (t + 3.0) + s, (s + u).max(0.0) + 0.25 * t + a * b * c]
}

#[kernel]
fn detached<N: AdNum>(x: [N; 2]) -> N {
    x[0].detach() * x[1]
}

fn lanes(v: V) -> [f64; W] {
    std::array::from_fn(|l| v.lane(l))
}

fn dot<const K: usize>(a: &[V; K], b: &[V; K]) -> V {
    a.iter().zip(b).fold(V::from(0.0), |s, (x, y)| s + *x * *y)
}

fn primal(x: &[V; 4], z: V) -> [V; 3] {
    all_ops::eval(x, &[z])
}

/// Jᵀ w by the compiled reverse sweep.
fn gradient(x: &[V; 4], z: V, w: &[V; 3]) -> [V; 4] {
    let mut a = [V::from(0.0); 4];
    all_ops::vjp(x, &[z], w, &mut a);
    a
}

/// max over lanes of |err| / (1 + |scale|)
fn worst(err: V, scale: V) -> f64 {
    lanes(err).iter().zip(lanes(scale)).map(|(e, s)| e.abs() / (1.0 + s.abs())).fold(0.0, f64::max)
}

#[test]
fn every_op() {
    let mut rng = StdRng::seed_from_u64(3);
    let mut randv = |lo: f64, hi: f64| V::from_fn(|_| rng.random_range(lo..hi));
    let (mut e_primal, mut e_dot, mut e_fd, mut e_hess, mut e_sym) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for _ in 0..20 {
        let x: [V; 4] = std::array::from_fn(|_| randv(0.2, 1.7)); // positive (ln, sqrt, powf)
        let v: [V; 4] = std::array::from_fn(|_| randv(-2.0, 2.0));
        let w: [V; 3] = std::array::from_fn(|_| randv(-2.0, 2.0));
        let z = randv(-2.0, 2.0);

        // 1. primal
        let y = primal(&x, z);
        for l in 0..W {
            let yr = reference(std::array::from_fn(|i| x[i].lane(l)), z.lane(l));
            for k in 0..3 {
                e_primal = e_primal.max((y[k].lane(l) - yr[k]).abs() / (1.0 + yr[k].abs()));
            }
        }

        // 2. dot test: forward tangent J v against reverse Jᵀ w
        let xd: [D; 4] = std::array::from_fn(|i| D::new(x[i], v[i]));
        let yd = all_ops::eval(&xd, &[D::new(z, V::from(0.0))]);
        let jv: [V; 3] = std::array::from_fn(|k| yd[k].t);
        let jtw = gradient(&x, z, &w);
        e_dot = e_dot.max(worst(dot(&w, &jv) - dot(&v, &jtw), dot(&w, &jv)));

        // 3. gradient vs central differences; output 2 holds the detach term (stop-gradient is
        //    not what finite differences measure), so it gets weight 0
        let mut w2 = w;
        w2[2] = V::from(0.0);
        let g = gradient(&x, z, &w2);
        for i in 0..4 {
            let h = x[i] * 1e-6;
            let (mut xp, mut xm) = (x, x);
            xp[i] += h;
            xm[i] -= h;
            let fd = (dot(&w2, &primal(&xp, z)) - dot(&w2, &primal(&xm, z))) / (h * 2.0);
            e_fd = e_fd.max(worst(g[i] - fd, fd));
        }

        // 4. Hessian of <w2, y> by forward-over-reverse vs central differences of the gradient
        let mut hess = [[V::from(0.0); 4]; 4];
        for j in 0..4 {
            let xj: [D; 4] = std::array::from_fn(|i| D::new(x[i], V::from(if i == j { 1.0 } else { 0.0 })));
            let sd: [D; 3] = std::array::from_fn(|k| D::new(w2[k], V::from(0.0)));
            let mut ad = [D::from(0.0); 4];
            all_ops::vjp(&xj, &[D::new(z, V::from(0.0))], &sd, &mut ad);
            let h = x[j] * 1e-6;
            let (mut xp, mut xm) = (x, x);
            xp[j] += h;
            xm[j] -= h;
            let (gp, gm) = (gradient(&xp, z, &w2), gradient(&xm, z, &w2));
            for i in 0..4 {
                let fd = (gp[i] - gm[i]) / (h * 2.0);
                hess[i][j] = ad[i].t;
                e_hess = e_hess.max(worst(ad[i].t - fd, fd));
            }
        }
        for (i, row) in hess.iter().enumerate() {
            for (j, &hij) in row.iter().enumerate().skip(i + 1) {
                e_sym = e_sym.max(worst(hij - hess[j][i], hij));
            }
        }
    }

    // 5. detach: y = stop(a) * b  =>  dy/da = 0, dy/db = a
    let mut ag = [V::from(0.0); 2];
    detached::vjp(&[V::from(1.7), V::from(0.3)], V::from(1.0), &mut ag);
    let e_detach = worst(ag[0], V::from(0.0)) + worst(ag[1] - 1.7, V::from(0.0));

    let checks = [
        ("primal vs reference", e_primal, 1e-13),
        ("dot test (forward vs reverse)", e_dot, 1e-13),
        ("gradient vs central differences", e_fd, 1e-6), // rounding-limited (h = 1e-6)
        ("Hessian vs FD of gradient", e_hess, 2e-5),
        ("Hessian symmetry", e_sym, 1e-12),
        ("detach (stop-gradient)", e_detach, 0.0),
    ];
    for (name, err, tol) in checks {
        println!("{name}: max err {err:e} (tol {tol:e})");
    }
    for (name, err, tol) in checks {
        assert!(err <= tol, "{name}: {err:e} > {tol:e}");
    }
}

#[test]
fn graph_matches_tracer() {
    let traced = trace(|x: [Tracer; 4], d: [Tracer; 1]| all_ops(x, d));
    assert_eq!(all_ops::graph(), traced);
    assert_eq!(traced.len(), 59);
    assert_eq!(all_ops::NRES, 18);
}

#[test]
fn eager_matches_compiled() {
    let x = [0.7, 1.3, 0.9, 1.1];
    let y = all_ops(x, [0.4]);
    let yc = all_ops::eval(&x.map(V::from), &[V::from(0.4)]);
    for k in 0..3 {
        assert!((y[k] - yc[k].lane(0)).abs() <= 1e-14 * y[k].abs(), "output {k}");
    }
}

/// `DualVec<_, 4>` seeded with the unit directions == four `Dual` runs, compiled (vjp: values,
/// output tangents, Hessian columns) and eager (each against its own kind: only compiled code
/// fuses FMAs). Values bit-equal; tangents may differ in the last bits (shared reciprocals).
#[test]
fn dual_vec_matches_duals() {
    type DV = DualVec<V, 4>;
    let mut rng = StdRng::seed_from_u64(5);
    let mut randv = |lo: f64, hi: f64| V::from_fn(|_| rng.random_range(lo..hi));
    let mut err = 0.0f64;
    for _ in 0..20 {
        let x: [V; 4] = std::array::from_fn(|_| randv(0.2, 1.7));
        let w: [V; 3] = std::array::from_fn(|_| randv(-2.0, 2.0));
        let z = randv(-1.0, 1.0);
        let xv: [DV; 4] = std::array::from_fn(|i| DV::unit(x[i], i));
        let mut av = [DV::from(0.0); 4];
        let yv = all_ops::vjp(&xv, &[DV::lift(z)], &w.map(DV::lift), &mut av);
        let ye = all_ops(xv, [DV::lift(z)]);
        for k in 0..4 {
            let xd: [D; 4] = std::array::from_fn(|i| D::new(x[i], V::from(if i == k { 1.0 } else { 0.0 })));
            let mut ad = [D::from(0.0); 4];
            let yd = all_ops::vjp(&xd, &[D::lift(z)], &w.map(D::lift), &mut ad);
            let yde = all_ops(xd, [D::lift(z)]);
            for o in 0..3 {
                assert_eq!(yv[o].v, yd[o].v, "value {o}");
                assert_eq!(ye[o].v, yde[o].v, "eager value {o}");
                err = err.max(worst(yv[o].t[k] - yd[o].t, yd[o].t)).max(worst(ye[o].t[k] - yde[o].t, yde[o].t));
            }
            for i in 0..4 {
                assert_eq!(av[i].v, ad[i].v, "gradient {i}");
                err = err.max(worst(av[i].t[k] - ad[i].t, ad[i].t));
            }
        }
    }
    println!("DualVec vs Dual: max tangent err {err:e}");
    assert!(err <= 1e-13, "{err:e}");
}
