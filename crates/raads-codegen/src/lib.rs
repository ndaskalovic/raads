//! Kernel graphs and the code generator behind `raads`.
//!
//! A kernel is a branch-free function traced (or parsed) into a [`Graph`]: inputs, data,
//! constants and operations in evaluation order, plus the output nodes. Loops stay loops:
//! their nodes sit between a [`Op::Loop`] marker and its [`Op::End`]. The passes in
//! [`passes`] decide which nodes need adjoints and which forward values the reverse sweep
//! reads; [`emit`] turns the graph into Rust generic over `R: raads::Real`.
//! Internal to `raads` and its macros: no stability guarantees.

#![warn(missing_docs)]

pub mod emit;
mod int;
pub mod passes;
mod unroll;

pub use emit::{Group, Signature};
pub use int::{Int, Opaque};
pub use unroll::unroll;

/// Operand slot that is not used.
pub const NONE: usize = usize::MAX;

/// Node operation. `Const` and `Powk` carry Rust source of an `f64` expression, so the
/// macro front end can keep e.g. `Self::EPS` symbolic; the tracer writes literals.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Differentiable input number `i` (counted over input groups in signature order).
    Input(usize),
    /// Non-differentiable input number `i` (counted over data groups).
    Data(usize),
    /// Constant (`f64` source).
    Const(String),
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
    /// `a / b`
    Div,
    /// `-a`
    Neg,
    /// `exp(a)`
    Exp,
    /// `ln(a)`
    Ln,
    /// `sqrt(a)`
    Sqrt,
    /// `a ** k`
    Powk(String),
    /// `max(a, 0)`
    Relu,
    /// `a > b ? c : d`; no derivative flows into `a`, `b`.
    SelGt,
    /// value of `a`, zero derivative
    Stop,
    /// `for var in lo..hi {`: the loop's nodes follow, up to the matching [`Op::End`].
    Loop(String, Int, Int),
    /// End of the innermost open loop.
    End,
    /// A value carried by the enclosing loop: `a` before the first iteration, the `b` of its
    /// [`Op::Next`] after each one. Sits at the start of the loop body.
    Phi,
    /// End of an iteration: the [`Op::Phi`] `a` takes the value `b`.
    Next,
    /// Value of the [`Op::Phi`] `a` after its loop. Sits right after the loop's [`Op::End`].
    Out,
    /// Local array `[a; len]`.
    Array(usize),
    /// Element `idx` of the local [`Op::Array`] `a`.
    Load(Int),
    /// Element `idx` of the local [`Op::Array`] `a` is set to `b`.
    Store(Int),
    /// A plain Rust `f64` expression of the loop variables (`i as f64`, `weight(h)`).
    Num(Opaque<f64>),
    /// `let p<i>: usize = idx;`: an integer local (`let k: usize = MATURITIES[i];`), computed
    /// once where it is defined; [`Int`]s read it as the variable `p<i>`.
    LetInt(Int),
    /// `let p<i>: f64 = ..;`: a plain Rust `f64` local of the loop variables, computed once
    /// where it is defined; plain Rust expressions read it as `p<i>`.
    LetNum(Opaque<f64>),
    /// Element `idx` of an array parameter.
    Param {
        /// Input (or data) number of element 0.
        first: usize,
        /// Array length.
        len: usize,
        /// A data parameter.
        data: bool,
        /// Element index.
        idx: Int,
    },
}

/// An operation and its operand node indices ([`NONE`] when unused).
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// Operation.
    pub op: Op,
    /// First operand.
    pub a: usize,
    /// Second operand.
    pub b: usize,
    /// Third operand.
    pub c: usize,
    /// Fourth operand.
    pub d: usize,
}

impl Node {
    /// Node from its parts.
    pub fn new(op: Op, a: usize, b: usize, c: usize, d: usize) -> Self {
        Node { op, a, b, c, d }
    }
    /// Operand indices actually used.
    pub fn args(&self) -> impl Iterator<Item = usize> + '_ {
        [self.a, self.b, self.c, self.d].into_iter().filter(|&i| i != NONE)
    }
    /// Input, data or constant.
    pub fn is_leaf(&self) -> bool {
        matches!(self.op, Op::Input(_) | Op::Data(_) | Op::Const(_))
    }
}

/// Nodes in evaluation order (operands always precede their users) and the output nodes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Graph {
    /// Nodes in evaluation order.
    pub nodes: Vec<Node>,
    /// Output node indices.
    pub outs: Vec<usize>,
}

impl Graph {
    /// Append a node; returns its index.
    pub fn push(&mut self, op: Op, a: usize, b: usize, c: usize, d: usize) -> usize {
        debug_assert!([a, b, c, d].iter().all(|&i| i == NONE || i < self.nodes.len()));
        self.nodes.push(Node::new(op, a, b, c, d));
        self.nodes.len() - 1
    }
    /// Innermost [`Op::Loop`] around each node ([`NONE`] at the top level). An
    /// [`Op::Loop`] belongs to the loop around it, an [`Op::End`] to the loop it closes.
    pub fn parents(&self) -> Vec<usize> {
        let mut open = Vec::new();
        let mut parent = vec![NONE; self.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            parent[i] = open.last().copied().unwrap_or(NONE);
            match n.op {
                Op::Loop(..) => open.push(i),
                Op::End => {
                    open.pop();
                }
                _ => {}
            }
        }
        parent
    }
    /// The matching [`Op::End`] of each [`Op::Loop`] ([`NONE`] for other nodes).
    pub fn ends(&self) -> Vec<usize> {
        let mut open = Vec::new();
        let mut end = vec![NONE; self.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            match n.op {
                Op::Loop(..) => open.push(i),
                Op::End => end[open.pop().expect("unbalanced loop")] = i,
                _ => {}
            }
        }
        end
    }
    /// The graph has loops.
    pub fn has_loops(&self) -> bool {
        self.nodes.iter().any(|n| matches!(n.op, Op::Loop(..)))
    }
    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    /// No nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Rust source for an `f64` literal that round-trips exactly.
pub fn f64_literal(k: f64) -> String {
    if k.is_nan() {
        "f64::NAN".into()
    } else if k.is_infinite() {
        if k > 0.0 { "f64::INFINITY".into() } else { "f64::NEG_INFINITY".into() }
    } else {
        let s = format!("{k:?}");
        if s.contains(['.', 'e', 'E']) { format!("{s}_f64") } else { format!("{s}.0_f64") }
    }
}
