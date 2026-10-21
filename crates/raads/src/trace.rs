//! Runtime tracer: runs a kernel's generic source on [`Tracer`] values and records the graph.
//!
//! Used as the oracle for `#[kernel]`: the graph the macro builds from tokens must equal
//! the graph recorded by executing the same function (node order:
//! inputs, data, then operations; `x / k` is recorded as `x * (1/k)`).

use crate::AdNum;
use raads_codegen::{Graph, NONE, Op, f64_literal};
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

thread_local! {
    static G: RefCell<Graph> = RefCell::new(Graph::default());
}

/// A graph node handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tracer(pub usize);

fn push(op: Op, a: usize, b: usize, c: usize, d: usize) -> Tracer {
    Tracer(G.with(|g| g.borrow_mut().push(op, a, b, c, d)))
}

fn unary(op: Op, a: Tracer) -> Tracer {
    push(op, a.0, NONE, NONE, NONE)
}

fn binary(op: Op, a: Tracer, b: Tracer) -> Tracer {
    push(op, a.0, b.0, NONE, NONE)
}

impl From<f64> for Tracer {
    fn from(k: f64) -> Tracer {
        push(Op::Const(f64_literal(k)), NONE, NONE, NONE, NONE)
    }
}

impl Add for Tracer {
    type Output = Tracer;
    fn add(self, r: Tracer) -> Tracer {
        binary(Op::Add, self, r)
    }
}
impl Sub for Tracer {
    type Output = Tracer;
    fn sub(self, r: Tracer) -> Tracer {
        binary(Op::Sub, self, r)
    }
}
impl Mul for Tracer {
    type Output = Tracer;
    fn mul(self, r: Tracer) -> Tracer {
        binary(Op::Mul, self, r)
    }
}
impl Div for Tracer {
    type Output = Tracer;
    fn div(self, r: Tracer) -> Tracer {
        binary(Op::Div, self, r)
    }
}
impl Neg for Tracer {
    type Output = Tracer;
    fn neg(self) -> Tracer {
        unary(Op::Neg, self)
    }
}
impl Add<f64> for Tracer {
    type Output = Tracer;
    fn add(self, k: f64) -> Tracer {
        self + Tracer::from(k)
    }
}
impl Sub<f64> for Tracer {
    type Output = Tracer;
    fn sub(self, k: f64) -> Tracer {
        self - Tracer::from(k)
    }
}
impl Mul<f64> for Tracer {
    type Output = Tracer;
    fn mul(self, k: f64) -> Tracer {
        self * Tracer::from(k)
    }
}
impl Div<f64> for Tracer {
    type Output = Tracer;
    fn div(self, k: f64) -> Tracer {
        self * Tracer::from(1.0 / k)
    }
}

impl crate::sealed::Sealed for Tracer {}

impl AdNum for Tracer {
    fn exp(self) -> Tracer {
        unary(Op::Exp, self)
    }
    fn ln(self) -> Tracer {
        unary(Op::Ln, self)
    }
    fn sqrt(self) -> Tracer {
        unary(Op::Sqrt, self)
    }
    fn powf(self, k: f64) -> Tracer {
        unary(Op::Powk(f64_literal(k)), self)
    }
    fn relu(self) -> Tracer {
        unary(Op::Relu, self)
    }
    fn detach(self) -> Tracer {
        unary(Op::Stop, self)
    }
    fn select_gt(a: Tracer, b: Tracer, x: Tracer, y: Tracer) -> Tracer {
        push(Op::SelGt, a.0, b.0, x.0, y.0)
    }
    fn rsub(self, k: f64) -> Tracer {
        Tracer::from(k) - self
    }
    fn rdiv(self, k: f64) -> Tracer {
        Tracer::from(k) / self
    }
    fn radd(self, k: f64) -> Tracer {
        Tracer::from(k) + self
    }
    fn rmul(self, k: f64) -> Tracer {
        Tracer::from(k) * self
    }
}

/// Trace `f` with `NI` differentiable inputs and `ND` data inputs (recorded in that order).
pub fn trace<const NI: usize, const ND: usize, const NO: usize>(f: impl FnOnce([Tracer; NI], [Tracer; ND]) -> [Tracer; NO]) -> Graph {
    G.with(|g| *g.borrow_mut() = Graph::default());
    let x = std::array::from_fn(|i| push(Op::Input(i), NONE, NONE, NONE, NONE));
    let d = std::array::from_fn(|i| push(Op::Data(i), NONE, NONE, NONE, NONE));
    let y = f(x, d);
    G.with(|g| {
        let mut g = std::mem::take(&mut *g.borrow_mut());
        g.outs = y.iter().map(|t| t.0).collect();
        g
    })
}
