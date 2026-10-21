//! Rust for one kernel graph, generic over `R: <krate>::Real`, where `krate` is the path of
//! the `raads` crate as the calling crate names it (usually `::raads`). Straight-line code,
//! except for the kernel's own loops.
//!
//! Four functions per kernel, all `#[inline(always)]`:
//! - `eval(inputs..) -> y`
//! - `vjp(inputs.., a_y, a_inputs..) -> y`: primal plus `a_inputs += Jᵀ a_y`
//! - `fwd(inputs.., res) -> y`: primal, saving the residuals `bwd` reads
//! - `bwd(data.., res, a_y, a_inputs..)`: `a_inputs += Jᵀ a_y` from residuals
//!
//! Rust never contracts `a * b + c`, so products are fused into `mul_add` explicitly:
//! in the forward pass when an `Add`/`Sub` consumes a `Mul`, and in the reverse pass a
//! product contribution stays pending until a second contribution (or the caller's
//! accumulator) can absorb it as one FMA. A constant factor stays an `f64` (`x * (k)`,
//! `x.mul_add_k(k, c)`), so `Dual` values never compute with a constant's zero tangent.
//!
//! Loops stay loops. Each forward iteration saves the values its reverse reads on a tape
//! (a `Vec` in `vjp`, `res` after the top-level residuals in `fwd`); the reverse loop runs
//! the iterations backwards and reads them back. Carried values `c<i>` have adjoints
//! `ac<i>`, local arrays `m<i>` have adjoint arrays `am<i>`; the adjoints of values defined
//! outside a loop and used in it are declared before the reverse loop and accumulate.

use crate::passes::Analysis;
use crate::{Graph, NONE, Op};
use std::fmt::Write;

/// One parameter of the kernel: `name: R` (len `None`) or `name: &[R; len]`.
#[derive(Clone, Debug)]
pub struct Group {
    /// Parameter name.
    pub name: String,
    /// Array length, `None` for a scalar.
    pub len: Option<usize>,
    /// Non-differentiable (no adjoint argument).
    pub data: bool,
}

/// Parameter groups in signature order and the output shape (`None`: a single `R`).
#[derive(Clone, Debug)]
pub struct Signature {
    /// Parameters.
    pub groups: Vec<Group>,
    /// Output array length, `None` for a scalar.
    pub out: Option<usize>,
}

impl Signature {
    fn param(&self, g: &Group) -> String {
        match g.len {
            None => format!("{}: R", g.name),
            Some(n) => format!("{}: &[R; {n}]", g.name),
        }
    }
    fn adj_param(&self, g: &Group) -> String {
        match g.len {
            None => format!("a_{}: &mut R", g.name),
            Some(n) => format!("a_{}: &mut [R; {n}]", g.name),
        }
    }
    fn out_ty(&self) -> String {
        match self.out {
            None => "R".into(),
            Some(n) => format!("[R; {n}]"),
        }
    }
    fn a_y_param(&self) -> String {
        match self.out {
            None => "a_y: R".into(),
            Some(n) => format!("a_y: &[R; {n}]"),
        }
    }
    fn a_y(&self, o: usize) -> String {
        match self.out {
            None => "a_y".into(),
            Some(_) => format!("a_y[{o}]"),
        }
    }
    /// Name of the array parameter whose element 0 is input (data) number `first`.
    fn group(&self, first: usize, data: bool) -> &str {
        let mut k = 0;
        for g in self.groups.iter().filter(|g| g.data == data) {
            if k == first {
                return &g.name;
            }
            k += g.len.unwrap_or(1);
        }
        panic!("no parameter starts at input {first} (data: {data})");
    }
    /// (value access, adjoint access) of differentiable input `i`, or data input `i`.
    fn slot(&self, i: usize, data: bool) -> (String, String) {
        let mut k = i;
        for g in self.groups.iter().filter(|g| g.data == data) {
            let n = g.len.unwrap_or(1);
            if k < n {
                return match g.len {
                    None => (g.name.clone(), format!("*a_{}", g.name)),
                    Some(_) => (format!("{}[{k}]", g.name), format!("a_{}[{k}]", g.name)),
                };
            }
            k -= n;
        }
        panic!("input {i} (data: {data}) outside the signature");
    }
}

/// Which of the four functions to emit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Primal only.
    Eval,
    /// Primal and reverse sweep.
    Vjp,
    /// Primal, saving residuals.
    Fwd,
    /// Reverse sweep from residuals.
    Bwd,
}

/// Number of residuals `fwd` saves: the top-level ones, then the loop tape.
pub fn nres(g: &Graph) -> usize {
    let an = Analysis::new(g);
    an.res.len() + if g.has_loops() { an.tape(g, false) } else { 0 }
}

/// Where the forward pass saves loop values and the reverse pass reads them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tape {
    /// Nowhere (`eval`).
    None,
    /// `tape: Vec<R>` (`vjp`).
    Vec,
    /// `res[tp..]` (`fwd`, `bwd`).
    Res,
}

/// Rust source of one function `name` (with visibility/qualifiers `vis`, e.g. `"pub"`);
/// `krate` is the path of the `raads` crate, e.g. `"::raads"`.
pub fn function(g: &Graph, sig: &Signature, kind: Kind, vis: &str, name: &str, krate: &str) -> String {
    let an = Analysis::new(g);
    let e = Emitter { g, sig, an: &an, krate };
    let mut params: Vec<String> = Vec::new();
    match kind {
        Kind::Eval | Kind::Vjp | Kind::Fwd => params.extend(sig.groups.iter().map(|g| sig.param(g))),
        Kind::Bwd => params.extend(sig.groups.iter().filter(|g| g.data).map(|g| sig.param(g))),
    }
    if kind == Kind::Fwd {
        params.push("res: &mut [R]".into());
    }
    if kind == Kind::Bwd {
        params.push("res: &[R]".into());
    }
    if matches!(kind, Kind::Vjp | Kind::Bwd) {
        params.push(sig.a_y_param());
        params.extend(sig.groups.iter().filter(|g| !g.data).map(|g| sig.adj_param(g)));
    }
    let ret = if kind == Kind::Bwd { String::new() } else { format!(" -> {}", sig.out_ty()) };
    let loops = g.has_loops();
    let doc = match kind {
        Kind::Eval => "The kernel's result.",
        Kind::Vjp => "The kernel's result; adds `Jᵀ a_y` to the input adjoints.",
        Kind::Fwd => "The kernel's result; saves the residuals `bwd` reads to `res`.",
        Kind::Bwd => "Adds `Jᵀ a_y` to the input adjoints, from the residuals `fwd` saved.",
    };
    let mut s = String::new();
    writeln!(
        s,
        "#[doc = {doc:?}]\n#[inline(always)]\n#[allow(unused_variables, unused_mut, clippy::all)]\n{vis} fn {name}<R: {krate}::Real>({}){ret} {{",
        params.join(", ")
    )
    .unwrap();
    // passive values are `R::Lanes`: their operations need the traits in scope
    writeln!(s, "    use {krate}::{{AdNum as _, Real as _}};").unwrap();
    match kind {
        Kind::Eval => {
            e.forward(&mut s, Tape::None);
            e.outputs(&mut s);
        }
        Kind::Vjp => {
            if loops {
                writeln!(s, "    let mut tape: ::std::vec::Vec<R> = ::std::vec::Vec::with_capacity({});", an.tape(g, true)).unwrap();
            }
            e.forward(&mut s, Tape::Vec);
            if loops {
                writeln!(s, "    let mut tp = tape.len();").unwrap();
            }
            e.reverse(&mut s, Tape::Vec);
            e.outputs(&mut s);
        }
        Kind::Fwd => {
            if loops {
                writeln!(s, "    let res = &mut res[..{}];", an.res.len() + an.tape(g, false)).unwrap();
                writeln!(s, "    let mut tp = {}usize;", an.res.len()).unwrap();
            }
            e.forward(&mut s, Tape::Res);
            if !an.res.is_empty() && !loops {
                writeln!(s, "    let res = &mut res[..{}];", an.res.len()).unwrap();
            }
            for (k, &r) in an.res.iter().enumerate() {
                writeln!(s, "    res[{k}] = {};", e.r(r)).unwrap();
            }
            e.outputs(&mut s);
        }
        Kind::Bwd => {
            if loops {
                let nres = an.res.len() + an.tape(g, false);
                writeln!(s, "    let res = &res[..{nres}];").unwrap();
                writeln!(s, "    let mut tp = {nres}usize;").unwrap();
            } else if !an.res.is_empty() {
                writeln!(s, "    let res = &res[..{}];", an.res.len()).unwrap();
            }
            for (k, &r) in an.res.iter().enumerate() {
                match an.active[r] {
                    true => writeln!(s, "    let v{r}: R = res[{k}];").unwrap(),
                    false => writeln!(s, "    let v{r}: R::Lanes = res[{k}].value();").unwrap(),
                }
            }
            for (i, n) in g.nodes.iter().enumerate() {
                if let Op::Data(j) = n.op
                    && an.reads[i]
                {
                    writeln!(s, "    let v{i}: R::Lanes = {}.value();", sig.slot(j, true).0).unwrap();
                }
            }
            e.reverse(&mut s, Tape::Res);
        }
    }
    s.push_str("}\n");
    s
}

/// A factor of a product: an active `R` expression, a passive `R::Lanes` value, or an `f64`
/// constant.
#[derive(Clone, Debug)]
enum Fac {
    R(String),
    L(String),
    K(String),
}

/// `x * y` (at least one factor active, or both constants).
fn prod(x: &Fac, y: &Fac) -> String {
    match (x, y) {
        (Fac::R(x), Fac::R(y)) => format!("{x} * {y}"),
        (Fac::R(x), Fac::L(l)) | (Fac::L(l), Fac::R(x)) => format!("{x}.mul_l({l})"),
        (Fac::R(x) | Fac::L(x), Fac::K(k)) | (Fac::K(k), Fac::R(x) | Fac::L(x)) => format!("{x} * ({k})"),
        (Fac::L(x), Fac::L(y)) => format!("{x} * {y}"),
        (Fac::K(k), Fac::K(l)) => format!("R::splat(({k}) * ({l}))"),
    }
}

/// `x * y + c`, fused (`c` an `R` expression). A passive product is lifted so that the FMA,
/// and with it the rounding, is the same as for plain packs.
fn fused(x: &Fac, y: &Fac, c: &str) -> String {
    match (x, y) {
        (Fac::R(x), Fac::R(y)) => format!("{x}.mul_add({y}, {c})"),
        (Fac::R(x), Fac::L(l)) | (Fac::L(l), Fac::R(x)) => format!("{x}.mul_add_l({l}, {c})"),
        (Fac::R(x), Fac::K(k)) | (Fac::K(k), Fac::R(x)) => format!("{x}.mul_add_k({k}, {c})"),
        (Fac::L(x), Fac::L(l)) => format!("R::lift({x}).mul_add_l({l}, {c})"),
        (Fac::L(x), Fac::K(k)) | (Fac::K(k), Fac::L(x)) => format!("R::lift({x}).mul_add_k({k}, {c})"),
        (Fac::K(k), Fac::K(l)) => format!("{c} + ({k}) * ({l})"),
    }
}

/// `c - x * y`, fused.
fn nfused(x: &Fac, y: &Fac, c: &str) -> String {
    match (x, y) {
        (Fac::R(x), Fac::R(y)) => format!("{x}.nmul_add({y}, {c})"),
        (Fac::R(x), Fac::L(l)) | (Fac::L(l), Fac::R(x)) => format!("{x}.nmul_add_l({l}, {c})"),
        (Fac::R(x), Fac::K(k)) | (Fac::K(k), Fac::R(x)) => format!("{x}.nmul_add_k({k}, {c})"),
        (Fac::L(x), Fac::L(l)) => format!("R::lift({x}).nmul_add_l({l}, {c})"),
        (Fac::L(x), Fac::K(k)) | (Fac::K(k), Fac::L(x)) => format!("R::lift({x}).nmul_add_k({k}, {c})"),
        (Fac::K(k), Fac::K(l)) => format!("{c} - ({k}) * ({l})"),
    }
}

struct Emitter<'a> {
    g: &'a Graph,
    sig: &'a Signature,
    an: &'a Analysis,
    krate: &'a str,
}

impl Emitter<'_> {
    fn konst(&self, i: usize) -> Option<&str> {
        match &self.g.nodes[i].op {
            Op::Const(k) => Some(k),
            _ => None,
        }
    }
    fn act(&self, i: usize) -> bool {
        self.an.active[i]
    }
    /// Value of node `i` as an `R` expression: constants splatted, passive values lifted.
    fn r(&self, i: usize) -> String {
        match self.konst(i) {
            Some(k) => format!("R::splat({k})"),
            None if self.act(i) => format!("v{i}"),
            None => format!("R::lift(v{i})"),
        }
    }
    /// Value of passive node `i` as an `R::Lanes` expression.
    fn l(&self, i: usize) -> String {
        match self.konst(i) {
            Some(k) => format!("<R::Lanes as ::core::convert::From<f64>>::from({k})"),
            None => format!("v{i}"),
        }
    }
    /// Node `i` as a product factor.
    fn fac(&self, i: usize) -> Fac {
        match self.konst(i) {
            Some(k) => Fac::K(k.into()),
            None if self.act(i) => Fac::R(format!("v{i}")),
            None => Fac::L(format!("v{i}")),
        }
    }
    /// `x * y` as a pair for `mul_add` on `R::Lanes` (both factors as `Fac::R`: in passive
    /// code every value is a pack), if node `i` is a passive product with a non-constant factor.
    fn passive_product(&self, i: usize) -> Option<(Fac, Fac)> {
        let n = &self.g.nodes[i];
        let f = |j: usize| match self.konst(j) {
            Some(k) => Fac::K(k.into()),
            None => Fac::R(format!("v{j}")),
        };
        (n.op == Op::Mul && !self.act(i) && !(self.konst(n.a).is_some() && self.konst(n.b).is_some())).then(|| (f(n.a), f(n.b)))
    }
    /// `x * y` as a pair for `mul_add`, if node `i` is a product with a non-constant factor.
    fn product(&self, i: usize) -> Option<(Fac, Fac)> {
        let n = &self.g.nodes[i];
        (n.op == Op::Mul && !(self.konst(n.a).is_some() && self.konst(n.b).is_some())).then(|| (self.fac(n.a), self.fac(n.b)))
    }

    /// `R` or `R::Lanes`, the type of node `i`.
    fn ty(&self, i: usize) -> &'static str {
        if self.act(i) { "R" } else { "R::Lanes" }
    }
    /// Value of node `j` as the type of node `i`.
    fn as_ty(&self, i: usize, j: usize) -> String {
        if self.act(i) { self.r(j) } else { self.l(j) }
    }
    /// The carried values of loop `lp`.
    fn phis(&self, lp: usize) -> impl Iterator<Item = usize> + '_ {
        (lp + 1..self.an.ends[lp]).filter(move |&p| self.an.parents[p] == lp && self.g.nodes[p].op == Op::Phi)
    }
    /// Element `idx` of an array parameter.
    fn param(&self, first: usize, data: bool, idx: &crate::Int) -> String {
        format!("{}[{idx}]", self.sig.group(first, data))
    }

    fn forward(&self, s: &mut String, tape: Tape) {
        for (i, n) in self.g.nodes.iter().enumerate() {
            if !self.an.live[i] {
                continue;
            }
            match &n.op {
                Op::Loop(var, lo, hi) => {
                    for p in self.phis(i).filter(|&p| self.an.live[p]) {
                        writeln!(s, "    let mut c{p}: {} = {};", self.ty(p), self.as_ty(p, self.g.nodes[p].a)).unwrap();
                    }
                    writeln!(s, "    for {var} in {lo}..{hi} {{").unwrap();
                    continue;
                }
                Op::End => {
                    let lp = self.an.parents[i];
                    if tape != Tape::None {
                        for r in self.an.saved(self.g, lp, tape == Tape::Vec) {
                            let v = self.r(r);
                            match tape {
                                Tape::Vec => writeln!(s, "    tape.push({v});").unwrap(),
                                _ => writeln!(s, "    res[tp] = {v};\n    tp += 1;").unwrap(),
                            }
                        }
                    }
                    writeln!(s, "    }}").unwrap();
                    continue;
                }
                Op::Phi => writeln!(s, "    let v{i}: {} = c{i};", self.ty(i)).unwrap(),
                Op::Next => writeln!(s, "    c{} = {};", n.a, self.as_ty(n.a, n.b)).unwrap(),
                Op::Out => writeln!(s, "    let v{i}: {} = c{};", self.ty(i), n.a).unwrap(),
                Op::Array(len) => writeln!(s, "    let mut m{i}: [{}; {len}] = [{}; {len}];", self.ty(i), self.as_ty(i, n.a)).unwrap(),
                Op::Load(idx) => writeln!(s, "    let v{i}: {} = m{}[{idx}];", self.ty(i), n.a).unwrap(),
                Op::Store(idx) => writeln!(s, "    m{}[{idx}] = {};", n.a, self.as_ty(n.a, n.b)).unwrap(),
                Op::Param { first, data, idx, .. } => writeln!(s, "    {}", self.load_param(i, *first, *data, idx)).unwrap(),
                Op::Int(idx) => writeln!(s, "    {}", Self::load_int(i, idx)).unwrap(),
                _ if self.act(i) => writeln!(s, "    let v{i}: R = {};", self.active_value(n)).unwrap(),
                _ => writeln!(s, "    let v{i}: R::Lanes = {};", self.passive_value(n)).unwrap(),
            }
        }
    }

    /// `let v<i> = ..;` for an integer as a number.
    fn load_int(i: usize, idx: &crate::Int) -> String {
        format!("let v{i}: R::Lanes = <R::Lanes as ::core::convert::From<f64>>::from({idx} as f64);")
    }
    /// `let v<i> = ..;` for an element of an array parameter.
    fn load_param(&self, i: usize, first: usize, data: bool, idx: &crate::Int) -> String {
        match data {
            true => format!("let v{i}: R::Lanes = {}.value();", self.param(first, data, idx)),
            false => format!("let v{i}: R = {};", self.param(first, data, idx)),
        }
    }

    /// A node that depends only on data and constants, as an `R::Lanes` expression.
    fn passive_value(&self, n: &crate::Node) -> String {
        let (a, b) = (n.a, n.b);
        let ka = if a != NONE { self.konst(a) } else { None };
        let kb = if b != NONE { self.konst(b) } else { None };
        match &n.op {
            Op::Input(_) => unreachable!("inputs are active"),
            Op::Data(j) => format!("{}.value()", self.sig.slot(*j, true).0),
            Op::Const(k) => self.l_const(k),
            Op::Add => match (ka, kb, self.passive_product(a), self.passive_product(b)) {
                (_, Some(k), _, _) => format!("v{a} + ({k})"),
                (Some(k), _, _, _) => format!("v{b} + ({k})"),
                (_, _, _, Some((x, y))) => fused(&x, &y, &format!("v{a}")),
                (_, _, Some((x, y)), _) => fused(&x, &y, &format!("v{b}")),
                _ => format!("v{a} + v{b}"),
            },
            Op::Sub => match (ka, kb, self.passive_product(b)) {
                (_, Some(k), _) => format!("v{a} - ({k})"),
                (Some(k), _, _) => format!("v{b}.rsub({k})"),
                (_, _, Some((x, y))) => nfused(&x, &y, &format!("v{a}")),
                _ => format!("v{a} - v{b}"),
            },
            Op::Mul => match (ka, kb) {
                (_, Some(k)) => format!("v{a} * ({k})"),
                (Some(k), _) => format!("v{b} * ({k})"),
                _ => format!("v{a} * v{b}"),
            },
            Op::Div => match (ka, kb) {
                (_, Some(k)) => format!("v{a} / ({k})"),
                (Some(k), _) => format!("v{b}.rdiv({k})"),
                _ => format!("v{a} / v{b}"),
            },
            Op::Neg => format!("-v{a}"),
            Op::Exp => format!("v{a}.exp()"),
            Op::Ln => format!("v{a}.ln()"),
            Op::Sqrt => format!("v{a}.sqrt()"),
            Op::Powk(k) => format!("v{a}.powf({k})"),
            Op::Relu => format!("v{a}.relu()"),
            Op::SelGt => {
                format!("<R::Lanes as {}::AdNum>::select_gt({}, {}, {}, {})", self.krate, self.l(a), self.l(b), self.l(n.c), self.l(n.d))
            }
            Op::Stop if self.act(a) => format!("v{a}.value()"),
            Op::Stop => format!("v{a}"),
            _ => unreachable!("loop operations are emitted by `forward`"),
        }
    }

    fn l_const(&self, k: &str) -> String {
        format!("<R::Lanes as ::core::convert::From<f64>>::from({k})")
    }

    /// A node that depends on a differentiable input, as an `R` expression; passive operands
    /// enter through the `_l` operations.
    fn active_value(&self, n: &crate::Node) -> String {
        let (a, b) = (n.a, n.b);
        let ka = if a != NONE { self.konst(a) } else { None };
        let kb = if b != NONE { self.konst(b) } else { None };
        let (pa, pb) = (a != NONE && !self.act(a), b != NONE && !self.act(b));
        match &n.op {
            Op::Input(j) => self.sig.slot(*j, false).0,
            Op::Add => match (ka, kb, self.product(a), self.product(b)) {
                (_, Some(k), _, _) => format!("v{a} + ({k})"),
                (Some(k), _, _, _) => format!("v{b} + ({k})"),
                (_, _, _, Some((x, y))) => fused(&x, &y, &self.r(a)),
                (_, _, Some((x, y)), _) => fused(&x, &y, &self.r(b)),
                _ if pb => format!("v{a}.add_l(v{b})"),
                _ if pa => format!("v{b}.add_l(v{a})"),
                _ => format!("v{a} + v{b}"),
            },
            Op::Sub => match (ka, kb, self.product(b)) {
                (_, Some(k), _) => format!("v{a} - ({k})"),
                (Some(k), _, _) => format!("v{b}.rsub({k})"),
                (_, _, Some((x, y))) => nfused(&x, &y, &self.r(a)),
                _ if pb => format!("v{a}.sub_l(v{b})"),
                _ if pa => format!("v{b}.l_sub(v{a})"),
                _ => format!("v{a} - v{b}"),
            },
            Op::Mul => prod(&self.fac(a), &self.fac(b)),
            Op::Div => match (ka, kb) {
                (_, Some(k)) => format!("v{a} / ({k})"),
                (Some(k), _) => format!("v{b}.rdiv({k})"),
                _ if pb => format!("v{a}.div_l(v{b})"),
                _ if pa => format!("v{b}.l_div(v{a})"),
                _ => format!("v{a} / v{b}"),
            },
            Op::Neg => format!("-v{a}"),
            Op::Exp => format!("v{a}.exp()"),
            Op::Ln => format!("v{a}.ln()"),
            Op::Sqrt => format!("v{a}.sqrt()"),
            Op::Powk(k) => format!("v{a}.powf({k})"),
            Op::Relu => format!("v{a}.relu()"),
            Op::SelGt => format!("R::select_gt({}, {}, {}, {})", self.r(a), self.r(b), self.r(n.c), self.r(n.d)),
            Op::Data(_) | Op::Const(_) | Op::Stop => unreachable!("leaves other than inputs, and detach, are passive"),
            _ => unreachable!("loop operations are emitted by `forward`"),
        }
    }

    fn outputs(&self, s: &mut String) {
        let vals: Vec<String> = self.g.outs.iter().map(|&o| self.r(o)).collect();
        match self.sig.out {
            None => writeln!(s, "    {}", vals[0]).unwrap(),
            Some(_) => writeln!(s, "    [{}]", vals.join(", ")).unwrap(),
        }
    }

    /// Adjoint arrays of the local arrays defined directly in loop `lp` (or at the top
    /// level: `NONE`), zero at the start of its reverse sweep.
    fn adjoint_arrays(&self, s: &mut String, lp: usize) {
        for (i, n) in self.g.nodes.iter().enumerate() {
            if let Op::Array(len) = n.op
                && self.an.parents[i] == lp
                && self.an.need[i]
            {
                writeln!(s, "    let mut am{i}: [R; {len}] = [R::neg_zero(); {len}];").unwrap();
            }
        }
    }

    /// Operands of node `n` that receive adjoint contributions from it.
    fn adjoint_operands(n: &crate::Node) -> Vec<usize> {
        match n.op {
            Op::SelGt => vec![n.c, n.d],
            Op::Next | Op::Store(_) => vec![n.b],
            Op::Out | Op::Load(_) | Op::Loop(..) | Op::End | Op::Param { .. } | Op::Int(_) => vec![],
            _ => n.args().collect(),
        }
    }

    fn reverse(&self, s: &mut String, tape: Tape) {
        let (g, need) = (self.g, &self.an.need);
        let mut ad = Adj { seen: vec![false; g.len()], pending: vec![None; g.len()], need };
        let buf = if tape == Tape::Vec { "tape" } else { "res" };
        self.adjoint_arrays(s, NONE);
        for (o, &i) in g.outs.iter().enumerate() {
            ad.add(s, i, self.sig.a_y(o), false);
        }
        for i in (0..g.len()).rev() {
            let n = &g.nodes[i];
            let (a, b) = (n.a, n.b);
            match &n.op {
                Op::End => {
                    let lp = self.an.parents[i];
                    let Op::Loop(var, lo, hi) = &g.nodes[lp].op else { unreachable!() };
                    for p in self.phis(lp).filter(|&p| need[p]) {
                        // the loop's `Out`s sit right after its `End`
                        let out = (i + 1..g.len()).take_while(|&o| g.nodes[o].op == Op::Out).find(|&o| g.nodes[o].a == p);
                        let init = match out {
                            Some(o) if ad.seen[o] => {
                                ad.materialise(s, o);
                                format!("a{o}")
                            }
                            _ => "R::neg_zero()".into(),
                        };
                        writeln!(s, "    let mut ac{p}: R = {init};").unwrap();
                    }
                    // adjoints of values from outside the loop accumulate over its iterations
                    let mut outer: Vec<usize> =
                        (lp + 1..i).flat_map(|j| Self::adjoint_operands(&g.nodes[j])).filter(|&j| j < lp && need[j]).collect();
                    outer.sort_unstable();
                    outer.dedup();
                    for j in outer {
                        ad.declare(s, j);
                    }
                    writeln!(s, "    for {var} in ({lo}..{hi}).rev() {{").unwrap();
                    let saved = self.an.saved(g, lp, tape == Tape::Vec);
                    if !saved.is_empty() {
                        writeln!(s, "    tp -= {};", saved.len()).unwrap();
                    }
                    for (k, &r) in saved.iter().enumerate() {
                        match self.act(r) {
                            true => writeln!(s, "    let v{r}: R = {buf}[tp + {k}];").unwrap(),
                            false => writeln!(s, "    let v{r}: R::Lanes = {buf}[tp + {k}].value();").unwrap(),
                        }
                    }
                    for r in self.an.rematerialised(g, lp, tape == Tape::Vec) {
                        match &g.nodes[r].op {
                            Op::Param { first, data, idx, .. } => writeln!(s, "    {}", self.load_param(r, *first, *data, idx)).unwrap(),
                            Op::Int(idx) => writeln!(s, "    {}", Self::load_int(r, idx)).unwrap(),
                            _ => unreachable!(),
                        }
                    }
                    self.adjoint_arrays(s, lp);
                    continue;
                }
                Op::Loop(..) => {
                    writeln!(s, "    }}").unwrap();
                    for p in self.phis(i).filter(|&p| need[p]) {
                        ad.add(s, g.nodes[p].a, format!("ac{p}"), false);
                    }
                    continue;
                }
                Op::Phi => {
                    if need[i] {
                        ad.materialise(s, i);
                        let v = if ad.seen[i] { format!("a{i}") } else { "R::neg_zero()".into() };
                        writeln!(s, "    ac{i} = {v};").unwrap();
                    }
                    continue;
                }
                Op::Next => {
                    if need[a] {
                        ad.add(s, b, format!("ac{a}"), false);
                    }
                    continue;
                }
                Op::Store(idx) => {
                    if need[a] {
                        ad.add(s, b, format!("am{a}[{idx}]"), false);
                        writeln!(s, "    am{a}[{idx}] = R::neg_zero();").unwrap();
                    }
                    continue;
                }
                Op::Array(_) => {
                    if need[i] {
                        ad.add(s, a, format!("am{i}.iter().fold(R::neg_zero(), |s, &x| s + x)"), false);
                    }
                    continue;
                }
                _ => {}
            }
            if !need[i] || !ad.seen[i] {
                continue;
            }
            let acc = match &n.op {
                Op::Input(j) => Some(self.sig.slot(*j, false).1),
                Op::Param { first, idx, .. } => Some(format!("a_{}", self.param(*first, false, idx))),
                Op::Load(idx) => Some(format!("am{a}[{idx}]")),
                _ => None,
            };
            if let Some(acc) = acc {
                match ad.pending[i].take() {
                    Some((x, y)) => writeln!(s, "    {acc} = {};", fused(&x, &y, &acc)).unwrap(),
                    None => writeln!(s, "    {acc} += a{i};").unwrap(),
                }
                continue;
            }
            ad.materialise(s, i);
            let ai = format!("a{i}");
            match &n.op {
                Op::Add => {
                    ad.add(s, a, ai.clone(), false);
                    ad.add(s, b, ai, false);
                }
                Op::Sub => {
                    ad.add(s, a, ai.clone(), false);
                    ad.add(s, b, ai, true);
                }
                Op::Mul => {
                    ad.add_mul(s, a, ai.clone(), self.fac(b));
                    ad.add_mul(s, b, ai, self.fac(a));
                }
                Op::Div => {
                    let q = match self.fac(b) {
                        Fac::K(k) => format!("{ai} / ({k})"),
                        Fac::L(l) => format!("{ai}.div_l({l})"),
                        Fac::R(r) => format!("{ai} / {r}"),
                    };
                    if need[b] {
                        // one division for both operands: ā_a += ā/b, ā_b -= (ā/b)·y
                        writeln!(s, "    let q{i}: R = {q};").unwrap();
                        ad.add(s, a, format!("q{i}"), false);
                        ad.add_mul(s, b, format!("(-q{i})"), Fac::R(format!("v{i}")));
                    } else {
                        ad.add(s, a, q, false);
                    }
                }
                Op::Neg => ad.add(s, a, ai, true),
                Op::Exp => ad.add_mul(s, a, ai, Fac::R(format!("v{i}"))),
                Op::Ln => ad.add(s, a, format!("{ai} / v{a}"), false),
                Op::Sqrt => ad.add(s, a, format!("{ai} * 0.5_f64 / v{i}"), false),
                Op::Powk(k) => ad.add_mul(s, a, ai, Fac::R(format!("(v{a}.powf(({k}) - 1.0_f64) * ({k}))"))),
                Op::Relu => ad.add(s, a, format!("R::select_gt(v{a}, R::splat(0.0_f64), {ai}, R::splat(0.0_f64))"), false),
                Op::SelGt => {
                    let (va, vb) = (self.r(a), self.r(b));
                    ad.add(s, n.c, format!("R::select_gt({va}, {vb}, {ai}, R::splat(0.0_f64))"), false);
                    ad.add(s, n.d, format!("R::select_gt({va}, {vb}, R::splat(0.0_f64), {ai})"), false);
                }
                _ => {}
            }
        }
    }
}

/// Adjoint bookkeeping: first touch defines `a{j}`; a first contribution that is a
/// product stays pending so the next one (or the input accumulator) fuses it into an FMA.
struct Adj<'a> {
    seen: Vec<bool>,
    pending: Vec<Option<(Fac, Fac)>>,
    need: &'a [bool],
}

impl Adj<'_> {
    /// Declare the adjoint of `j` now (zero if it has no contribution yet), so later
    /// contributions, e.g. from inside a loop, accumulate into it.
    fn declare(&mut self, s: &mut String, j: usize) {
        if self.seen[j] {
            self.materialise(s, j);
        } else {
            self.seen[j] = true;
            writeln!(s, "    let mut a{j}: R = R::neg_zero();").unwrap();
        }
    }
    fn materialise(&mut self, s: &mut String, j: usize) {
        if let Some((x, y)) = self.pending[j].take() {
            writeln!(s, "    let mut a{j}: R = {};", prod(&x, &y)).unwrap();
        }
    }
    fn add_mul(&mut self, s: &mut String, j: usize, x: String, y: Fac) {
        if !self.need[j] {
            return;
        }
        let x = Fac::R(x);
        if !self.seen[j] {
            self.seen[j] = true;
            self.pending[j] = Some((x, y));
            return;
        }
        self.materialise(s, j);
        writeln!(s, "    a{j} = {};", fused(&x, &y, &format!("a{j}"))).unwrap();
    }
    fn add(&mut self, s: &mut String, j: usize, e: String, neg: bool) {
        if !self.need[j] {
            return;
        }
        if !self.seen[j] {
            self.seen[j] = true;
            writeln!(s, "    let mut a{j}: R = {}{e};", if neg { "-" } else { "" }).unwrap();
            return;
        }
        if neg {
            if let Some((x, y)) = self.pending[j].take() {
                writeln!(s, "    let mut a{j}: R = {} - ({e});", prod(&x, &y)).unwrap();
            } else {
                writeln!(s, "    a{j} -= {e};").unwrap();
            }
        } else if let Some((x, y)) = self.pending[j].take() {
            writeln!(s, "    let mut a{j}: R = {};", fused(&x, &y, &format!("({e})"))).unwrap();
        } else {
            writeln!(s, "    a{j} += {e};").unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_path() {
        let mut g = Graph::default();
        let x = g.push(Op::Input(0), NONE, NONE, NONE, NONE);
        let y = g.push(Op::Exp, x, NONE, NONE, NONE);
        g.outs.push(y);
        let sig = Signature { groups: vec![Group { name: "x".into(), len: None, data: false }], out: None };
        let src = function(&g, &sig, Kind::Eval, "pub", "f", "::aad");
        assert!(src.contains("pub fn f<R: ::aad::Real>(x: R) -> R {"), "{src}");
        assert!(!src.contains("raads"), "{src}");
    }

    /// `x / y`: the reverse sweep divides once (`q = ā/y`), shared by both adjoints.
    #[test]
    fn div_shares_quotient() {
        let mut g = Graph::default();
        let x = g.push(Op::Input(0), NONE, NONE, NONE, NONE);
        let y = g.push(Op::Input(1), NONE, NONE, NONE, NONE);
        let q = g.push(Op::Div, x, y, NONE, NONE);
        g.outs.push(q);
        let sig = Signature { groups: vec![Group { name: "x".into(), len: Some(2), data: false }], out: None };
        let src = function(&g, &sig, Kind::Vjp, "pub", "f", "::raads");
        assert_eq!(src.matches(" / ").count(), 2, "forward division plus one in the reverse sweep:\n{src}");
        assert!(src.contains("let q2: R = a2 / v1;"), "{src}");
        assert!(src.contains("(-q2).mul_add(v2, a_x[1])"), "{src}");
    }

    /// GBM step `s + c0 + c1 * z`: the data `z` is a passive pack, so `c1 * z` fuses with an
    /// `_l` operation and the reverse sweep scales by `z` without a tangent.
    #[test]
    fn passive_data() {
        let mut g = Graph::default();
        let s0 = g.push(Op::Input(0), NONE, NONE, NONE, NONE);
        let c0 = g.push(Op::Input(1), NONE, NONE, NONE, NONE);
        let c1 = g.push(Op::Input(2), NONE, NONE, NONE, NONE);
        let z = g.push(Op::Data(0), NONE, NONE, NONE, NONE);
        let a = g.push(Op::Add, s0, c0, NONE, NONE);
        let m = g.push(Op::Mul, c1, z, NONE, NONE);
        let y = g.push(Op::Add, a, m, NONE, NONE);
        g.outs.push(y);
        let sig = Signature {
            groups: vec![Group { name: "x".into(), len: Some(3), data: false }, Group { name: "z".into(), len: None, data: true }],
            out: None,
        };
        let src = function(&g, &sig, Kind::Vjp, "pub", "f", "::raads");
        assert!(src.contains("let v3: R::Lanes = z.value();"), "{src}");
        assert!(src.contains("v2.mul_add_l(v3, v4)"), "{src}");
        assert!(src.contains("a_x[2] = a5.mul_add_l(v3, a_x[2]);"), "{src}");
    }
}
