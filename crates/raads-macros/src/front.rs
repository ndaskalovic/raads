//! Kernel source → graph.
//!
//! Interprets the restricted kernel language symbolically, in Rust evaluation order, and
//! records the same nodes the runtime tracer records when it executes the function:
//! inputs, data, then operations. Plain `f64` subexpressions (literals, constants such as
//! `Self::EPS`) are not nodes; literals fold at expansion time, other expressions stay
//! symbolic and are evaluated by the compiler in the generated code.
//!
//! It also returns the function body rewritten so it compiles for any `N: AdNum`
//! (`2 * x` becomes `x.rmul(2f64)`, `x.maximum(0.0)` becomes `x.maximum(N::from(0.0))`),
//! which is the eager / tracer version of the kernel.

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{ToTokens, quote, quote_spanned};
use raads_codegen::{Graph, NONE, Op, f64_literal};
use std::collections::HashMap;
use syn::spanned::Spanned;
use syn::{BinOp, Error, Expr, Lit, Pat, Result, Stmt, UnOp};

/// An `f64` that is not a graph node.
#[derive(Clone, Debug)]
pub enum F {
    Num(f64),
    Expr(TokenStream),
}

impl F {
    /// Rust source of the value, for the emitter.
    pub fn source(&self) -> String {
        match self {
            F::Num(k) => f64_literal(*k),
            F::Expr(t) => format!("({t})"),
        }
    }
    /// Source as tokens that are safe in any position, including a method receiver
    /// (`(-1.0_f64).exp()`, not `-1.0_f64.exp()`).
    fn tokens(&self) -> TokenStream {
        match self {
            F::Num(k) if k.is_sign_negative() => format!("({})", self.source()).parse().unwrap(),
            _ => self.source().parse().unwrap(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Val {
    Node(usize),
    F(F),
    Arr(Vec<Val>),
}

/// Shape of a parameter or result: `N` or `[N; len]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Scalar,
    Array(usize),
}

pub struct Param {
    pub name: String,
    pub shape: Shape,
    pub data: bool,
}

pub struct Builder {
    pub g: Graph,
    /// Rust source of each constant node's value (as an `f64` expression).
    pub consts: Vec<Option<F>>,
    env: HashMap<String, Val>,
    /// The kernel's number type parameter and the path of `raads`.
    n: Ident,
    krate: TokenStream,
}

fn err<T>(span: Span, msg: &str) -> Result<T> {
    Err(Error::new(span, msg))
}

impl Builder {
    /// Records input nodes (non-data params in order), then data nodes.
    pub fn new(params: &[Param], n: &Ident, krate: &TokenStream) -> Self {
        let mut b = Builder { g: Graph::default(), consts: Vec::new(), env: HashMap::new(), n: n.clone(), krate: krate.clone() };
        for data in [false, true] {
            let mut k = 0;
            for p in params.iter().filter(|p| p.data == data) {
                let mut leaf = || {
                    let op = if data { Op::Data(k) } else { Op::Input(k) };
                    k += 1;
                    Val::Node(b.push(op, NONE, NONE, NONE, NONE))
                };
                let v = match p.shape {
                    Shape::Scalar => leaf(),
                    Shape::Array(n) => Val::Arr((0..n).map(|_| leaf()).collect()),
                };
                b.env.insert(p.name.clone(), v);
            }
        }
        b
    }

    fn push(&mut self, op: Op, a: usize, b: usize, c: usize, d: usize) -> usize {
        self.consts.push(None);
        self.g.push(op, a, b, c, d)
    }

    fn konst(&mut self, f: &F) -> usize {
        let i = self.push(Op::Const(f.source()), NONE, NONE, NONE, NONE);
        self.consts[i] = Some(f.clone());
        i
    }

    fn node(&mut self, v: &Val, span: Span) -> Result<usize> {
        match v {
            Val::Node(i) => Ok(*i),
            Val::F(f) => Ok(self.konst(f)),
            Val::Arr(_) => err(span, "expected a number, found an array"),
        }
    }

    /// Rewritten source `t` of `v` where an `N` is expected: plain `f64`s become `N::from(t)`
    /// (which the tracer records as the constant node `node` makes).
    fn as_n(&self, v: &Val, t: TokenStream) -> TokenStream {
        let n = &self.n;
        match v {
            Val::F(_) => quote!(<#n as ::core::convert::From<f64>>::from(#t)),
            _ => t,
        }
    }

    fn unary(&mut self, op: Op, a: usize) -> Val {
        Val::Node(self.push(op, a, NONE, NONE, NONE))
    }

    fn binary(&mut self, op: Op, a: usize, b: usize) -> Val {
        Val::Node(self.push(op, a, b, NONE, NONE))
    }

    /// Statements then the tail expression; returns its value and the rewritten body.
    pub fn block(&mut self, block: &syn::Block) -> Result<(Val, TokenStream)> {
        let mut out = TokenStream::new();
        let n = block.stmts.len();
        for (i, st) in block.stmts.iter().enumerate() {
            match st {
                Stmt::Local(l) => {
                    let init = l.init.as_ref().ok_or_else(|| Error::new(l.span(), "`let` needs a value"))?;
                    if init.diverge.is_some() {
                        return err(l.span(), "`let ... else` is not supported in a kernel");
                    }
                    let (v, e) = self.expr(&init.expr)?;
                    self.bind(&l.pat, v)?;
                    let pat = &l.pat;
                    out.extend(quote_spanned!(l.span()=> let #pat = #e;));
                }
                Stmt::Expr(e, semi) if i + 1 == n && semi.is_none() => {
                    let (v, t) = self.expr(e)?;
                    out.extend(t);
                    return Ok((v, out));
                }
                Stmt::Expr(Expr::Assign(a), Some(_)) => {
                    let name = ident_of(&a.left)?;
                    if !self.env.contains_key(&name) {
                        return err(a.left.span(), &format!("unknown variable `{name}`"));
                    }
                    let (v, e) = self.expr(&a.right)?;
                    self.env.insert(name, v);
                    let l = &a.left;
                    out.extend(quote!(#l = #e;));
                }
                _ => return err(st.span(), "kernel statements are `let` bindings, assignments and a final value"),
            }
        }
        err(block.span(), "a kernel must end with its result value")
    }

    fn bind(&mut self, pat: &Pat, v: Val) -> Result<()> {
        match (pat, v) {
            (Pat::Ident(p), v) => {
                self.env.insert(p.ident.to_string(), v);
                Ok(())
            }
            (Pat::Type(t), v) => self.bind(&t.pat, v),
            (Pat::Wild(_), _) => Ok(()),
            (Pat::Tuple(t), Val::Arr(vs)) if t.elems.len() == vs.len() => t.elems.iter().zip(vs).try_for_each(|(p, v)| self.bind(p, v)),
            (Pat::Slice(s), Val::Arr(vs)) if s.elems.len() == vs.len() => s.elems.iter().zip(vs).try_for_each(|(p, v)| self.bind(p, v)),
            (p, _) => err(p.span(), "unsupported pattern (use a name, `_`, or a tuple/array of them)"),
        }
    }

    /// Value of `e` and its rewritten source.
    fn expr(&mut self, e: &Expr) -> Result<(Val, TokenStream)> {
        let sp = e.span();
        match e {
            Expr::Lit(l) => {
                let k = lit_f64(&l.lit)?;
                // kernel numbers are f64: `2` is written as `2f64` in the rewritten body
                let t = match &l.lit {
                    Lit::Int(i) => {
                        let mut f = Literal::f64_suffixed(k);
                        f.set_span(i.span());
                        f.into_token_stream()
                    }
                    _ => e.to_token_stream(),
                };
                Ok((Val::F(F::Num(k)), t))
            }
            Expr::Paren(p) => {
                let (v, t) = self.expr(&p.expr)?;
                Ok((v, quote!((#t))))
            }
            Expr::Group(g) => self.expr(&g.expr),
            Expr::Path(p) => {
                // A bare name is a variable, or an `f64` constant if it is SCREAMING_CASE;
                // paths (`Self::EPS`, `f64::consts::PI`) are `f64` constants.
                if let Some(id) = p.path.get_ident() {
                    let name = id.to_string();
                    if let Some(v) = self.env.get(&name) {
                        return Ok((v.clone(), e.to_token_stream()));
                    }
                    if !is_constant_name(&name) {
                        return err(sp, &format!("unknown variable `{name}` (constants are SCREAMING_CASE or paths)"));
                    }
                }
                Ok((Val::F(F::Expr(e.to_token_stream())), e.to_token_stream()))
            }
            Expr::Index(ix) => {
                let (v, t) = self.expr(&ix.expr)?;
                let Expr::Lit(syn::ExprLit { lit: Lit::Int(k), .. }) = &*ix.index else {
                    return err(ix.index.span(), "array indices in a kernel must be integer literals");
                };
                let k: usize = k.base10_parse()?;
                let Val::Arr(vs) = v else { return err(sp, "indexing a value that is not an array") };
                let v = vs.get(k).cloned().ok_or_else(|| Error::new(ix.index.span(), "index out of bounds"))?;
                let i = &ix.index;
                Ok((v, quote!(#t[#i])))
            }
            Expr::Array(a) => {
                let mut vs = Vec::new();
                let mut ts = Vec::new();
                for el in &a.elems {
                    let (v, t) = self.expr(el)?;
                    vs.push(v);
                    ts.push(t);
                }
                Ok((Val::Arr(vs), quote!([#(#ts),*])))
            }
            Expr::Tuple(a) => {
                let mut vs = Vec::new();
                let mut ts = Vec::new();
                for el in &a.elems {
                    let (v, t) = self.expr(el)?;
                    vs.push(v);
                    ts.push(t);
                }
                Ok((Val::Arr(vs), quote!((#(#ts,)*))))
            }
            Expr::Unary(u) if matches!(u.op, UnOp::Neg(_)) => {
                let (v, t) = self.expr(&u.expr)?;
                let v = match v {
                    Val::F(F::Num(k)) => Val::F(F::Num(-k)),
                    Val::F(f) => {
                        let t = f.tokens();
                        Val::F(F::Expr(quote!(-#t)))
                    }
                    v => {
                        let a = self.node(&v, sp)?;
                        self.unary(Op::Neg, a)
                    }
                };
                Ok((v, quote!(-#t)))
            }
            Expr::Binary(b) => self.binary_expr(b),
            Expr::MethodCall(m) => self.method(m),
            Expr::Call(c) => self.call(c),
            Expr::If(_) | Expr::Match(_) => err(sp, "a kernel is branch-free: use `N::select_gt(a, b, x, y)`"),
            _ => err(sp, "unsupported expression in a kernel"),
        }
    }

    fn binary_expr(&mut self, b: &syn::ExprBinary) -> Result<(Val, TokenStream)> {
        let sp = b.span();
        let (l, lt) = self.expr(&b.left)?;
        let (r, rt) = self.expr(&b.right)?;
        let (op, sym, rmeth) = match b.op {
            BinOp::Add(_) => (Op::Add, quote!(+), quote!(radd)),
            BinOp::Sub(_) => (Op::Sub, quote!(-), quote!(rsub)),
            BinOp::Mul(_) => (Op::Mul, quote!(*), quote!(rmul)),
            BinOp::Div(_) => (Op::Div, quote!(/), quote!(rdiv)),
            _ => return err(sp, "kernels support + - * / only (comparisons: `N::select_gt`)"),
        };
        match (l, r) {
            (Val::F(a), Val::F(c)) => {
                let v = match (&a, &c) {
                    (F::Num(x), F::Num(y)) => F::Num(match op {
                        Op::Add => x + y,
                        Op::Sub => x - y,
                        Op::Mul => x * y,
                        _ => x / y,
                    }),
                    _ => {
                        let (x, y) = (a.tokens(), c.tokens());
                        F::Expr(quote!(#x #sym #y))
                    }
                };
                Ok((Val::F(v), quote!(#lt #sym #rt)))
            }
            (Val::F(k), r) => {
                let x = self.node(&r, b.right.span())?;
                let c = self.konst(&k);
                let v = self.binary(op, c, x);
                let krate = &self.krate;
                Ok((v, quote!(#krate::AdNum::#rmeth(#rt, #lt))))
            }
            (l, Val::F(k)) => {
                let x = self.node(&l, b.left.span())?;
                let v = if op == Op::Div {
                    let inv = match &k {
                        F::Num(y) => F::Num(1.0 / y),
                        F::Expr(t) => F::Expr(quote!(1.0_f64 / (#t))),
                    };
                    let c = self.konst(&inv);
                    self.binary(Op::Mul, x, c)
                } else {
                    let c = self.konst(&k);
                    self.binary(op, x, c)
                };
                Ok((v, quote!(#lt #sym #rt)))
            }
            (l, r) => {
                let x = self.node(&l, b.left.span())?;
                let y = self.node(&r, b.right.span())?;
                Ok((self.binary(op, x, y), quote!(#lt #sym #rt)))
            }
        }
    }

    fn method(&mut self, m: &syn::ExprMethodCall) -> Result<(Val, TokenStream)> {
        let sp = m.span();
        let name = m.method.to_string();
        let (recv, rt) = self.expr(&m.receiver)?;
        let mut args = Vec::new();
        let mut ats = Vec::new();
        for a in &m.args {
            let (v, t) = self.expr(a)?;
            args.push(v);
            ats.push(t);
        }
        let meth = &m.method;
        if let Val::F(f) = &recv {
            if args.iter().any(|a| !matches!(a, Val::F(_))) {
                let n = &self.n;
                return err(m.receiver.span(), &format!("`{name}` on a plain f64: write `{n}::from(k).{name}(..)`"));
            }
            // plain f64 arithmetic, evaluated by the compiler; literal arguments keep their
            // type (`powi(2)` takes an `i32`)
            let ft = f.tokens();
            let fa = args.iter().zip(&m.args).map(|(v, a)| match v {
                Val::F(_) if is_literal(a) => a.to_token_stream(),
                Val::F(x) => x.tokens(),
                _ => unreachable!(),
            });
            let sa = ats.iter().zip(&m.args).map(|(t, a)| if is_literal(a) { a.to_token_stream() } else { t.clone() });
            let src = quote!(#rt.#meth(#(#sa),*));
            return Ok((Val::F(F::Expr(quote!(#ft.#meth(#(#fa),*)))), src));
        }
        let mut src = quote!(#rt.#meth(#(#ats),*));
        let x = self.node(&recv, m.receiver.span())?;
        let fk = |i: usize| -> Result<F> {
            match args.get(i) {
                Some(Val::F(f)) => Ok(f.clone()),
                _ => err(sp, &format!("`{name}` takes a plain f64 argument")),
            }
        };
        let arity = |n: usize| if args.len() == n { Ok(()) } else { err(sp, &format!("`{name}` takes {n} argument(s)")) };
        let v = match name.as_str() {
            "exp" | "ln" | "sqrt" | "relu" | "detach" => {
                arity(0)?;
                let op = match name.as_str() {
                    "exp" => Op::Exp,
                    "ln" => Op::Ln,
                    "sqrt" => Op::Sqrt,
                    "relu" => Op::Relu,
                    _ => Op::Stop,
                };
                self.unary(op, x)
            }
            "powf" => {
                arity(1)?;
                let k = fk(0)?;
                self.unary(Op::Powk(k.source()), x)
            }
            "radd" | "rsub" | "rmul" | "rdiv" => {
                arity(1)?;
                let c = self.konst(&fk(0)?);
                let op = match name.as_str() {
                    "radd" => Op::Add,
                    "rsub" => Op::Sub,
                    "rmul" => Op::Mul,
                    _ => Op::Div,
                };
                self.binary(op, c, x)
            }
            "maximum" | "minimum" => {
                arity(1)?;
                let y = self.node(&args[0], sp)?;
                let b = self.as_n(&args[0], ats[0].clone());
                src = quote!(#rt.#meth(#b));
                let (p, q) = if name == "maximum" { (x, y) } else { (y, x) };
                Val::Node(self.push(Op::SelGt, x, y, p, q))
            }
            "smooth_relu" => {
                // same node order as `AdNum::smooth_relu`
                arity(1)?;
                let eps = fk(0)?;
                let zero = self.konst(&F::Num(0.0));
                let nx = self.push(Op::Neg, x, NONE, NONE, NONE);
                let ax = self.push(Op::SelGt, x, zero, x, nx);
                let relu = self.push(Op::Relu, x, NONE, NONE, NONE);
                let nax = self.push(Op::Neg, ax, NONE, NONE, NONE);
                let inv = match &eps {
                    F::Num(y) => F::Num(1.0 / y),
                    F::Expr(t) => F::Expr(quote!(1.0_f64 / (#t))),
                };
                let ci = self.konst(&inv);
                let q = self.push(Op::Mul, nax, ci, NONE, NONE);
                let ex = self.push(Op::Exp, q, NONE, NONE, NONE);
                let one = self.konst(&F::Num(1.0));
                let s = self.push(Op::Add, one, ex, NONE, NONE);
                let l = self.push(Op::Ln, s, NONE, NONE, NONE);
                let ce = self.konst(&eps);
                let m = self.push(Op::Mul, ce, l, NONE, NONE);
                self.binary(Op::Add, relu, m)
            }
            _ => {
                return err(
                    m.method.span(),
                    "unsupported method in a kernel (exp ln sqrt powf relu detach maximum minimum smooth_relu radd rsub rmul rdiv)",
                );
            }
        };
        Ok((v, src))
    }

    fn call(&mut self, c: &syn::ExprCall) -> Result<(Val, TokenStream)> {
        let sp = c.span();
        let Expr::Path(p) = &*c.func else { return err(sp, "unsupported call in a kernel") };
        let last = p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
        let f = &c.func;
        if last == "select_gt" && c.args.len() == 4 {
            // a plain f64 argument becomes its constant node right after it is evaluated, as
            // `N::from(k)` does in the rewritten source
            let mut ids = Vec::new();
            let mut ats = Vec::new();
            for a in &c.args {
                let (v, t) = self.expr(a)?;
                ids.push(self.node(&v, a.span())?);
                ats.push(self.as_n(&v, t));
            }
            return Ok((Val::Node(self.push(Op::SelGt, ids[0], ids[1], ids[2], ids[3])), quote!(#f(#(#ats),*))));
        }
        let mut args = Vec::new();
        let mut ats = Vec::new();
        for a in &c.args {
            let (v, t) = self.expr(a)?;
            args.push(v);
            ats.push(t);
        }
        let src = quote!(#f(#(#ats),*));
        match (last.as_str(), args.len()) {
            ("from", 1) => match &args[0] {
                Val::F(k) => {
                    let k = k.clone();
                    Ok((Val::Node(self.konst(&k)), src))
                }
                _ => err(sp, "`N::from` takes a plain f64"),
            },
            _ => err(sp, "unsupported call in a kernel (`N::select_gt(a, b, x, y)`, `N::from(k)`)"),
        }
    }

    /// Graph outputs from the result value, checked against the declared shape.
    pub fn outputs(&mut self, v: &Val, shape: Shape, span: Span) -> Result<()> {
        let outs = match (shape, v) {
            (Shape::Scalar, v) => vec![self.node(v, span)?],
            (Shape::Array(n), Val::Arr(vs)) if vs.len() == n => vs.iter().map(|x| self.node(x, span)).collect::<Result<_>>()?,
            _ => return err(span, "result does not match the declared return type"),
        };
        self.g.outs = outs;
        Ok(())
    }

    /// Code that rebuilds the graph at run time (constants evaluated by the compiler),
    /// for comparison with the tracer.
    pub fn graph_code(&self) -> TokenStream {
        let krate = &self.krate;
        let nodes = self.g.nodes.iter().enumerate().map(|(i, n)| {
            let arg = |x: usize| if x == NONE { quote!(#krate::__private::codegen::NONE) } else { quote!(#x) };
            let (a, b, c, d) = (arg(n.a), arg(n.b), arg(n.c), arg(n.d));
            let lit = |f: &F| {
                let t = f.tokens();
                quote!(#krate::__private::codegen::f64_literal(#t))
            };
            let op = match &n.op {
                Op::Input(k) => quote!(Input(#k)),
                Op::Data(k) => quote!(Data(#k)),
                Op::Const(_) => {
                    let l = lit(self.consts[i].as_ref().unwrap());
                    quote!(Const(#l))
                }
                Op::Powk(s) => {
                    let t: TokenStream = s.parse().unwrap();
                    let l = quote!(#krate::__private::codegen::f64_literal(#t));
                    quote!(Powk(#l))
                }
                Op::Add => quote!(Add),
                Op::Sub => quote!(Sub),
                Op::Mul => quote!(Mul),
                Op::Div => quote!(Div),
                Op::Neg => quote!(Neg),
                Op::Exp => quote!(Exp),
                Op::Ln => quote!(Ln),
                Op::Sqrt => quote!(Sqrt),
                Op::Relu => quote!(Relu),
                Op::SelGt => quote!(SelGt),
                Op::Stop => quote!(Stop),
            };
            quote!(g.push(#krate::__private::codegen::Op::#op, #a, #b, #c, #d);)
        });
        let outs = &self.g.outs;
        quote! {
            let mut g = #krate::__private::codegen::Graph::default();
            #(#nodes)*
            g.outs = vec![#(#outs),*];
            g
        }
    }
}

fn ident_of(e: &Expr) -> Result<String> {
    match e {
        Expr::Path(p) if p.path.get_ident().is_some() => Ok(p.path.get_ident().unwrap().to_string()),
        _ => err(e.span(), "only plain variables can be assigned in a kernel"),
    }
}

/// `CONST`, `PI`, `EPS_2`: names that can only be constants, not misspelt variables.
fn is_constant_name(s: &str) -> bool {
    s.chars().any(|c| c.is_ascii_uppercase()) && s.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// A literal, possibly negated.
fn is_literal(e: &Expr) -> bool {
    match e {
        Expr::Lit(_) => true,
        Expr::Unary(u) if matches!(u.op, UnOp::Neg(_)) => is_literal(&u.expr),
        Expr::Paren(p) => is_literal(&p.expr),
        Expr::Group(g) => is_literal(&g.expr),
        _ => false,
    }
}

fn lit_f64(l: &Lit) -> Result<f64> {
    match l {
        Lit::Float(f) => f.base10_parse(),
        Lit::Int(i) => i.base10_parse(),
        _ => err(l.span(), "only numeric literals are allowed in a kernel"),
    }
}
