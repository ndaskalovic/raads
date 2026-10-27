//! Kernel source → graph.
//!
//! Interprets the restricted kernel language symbolically, in Rust evaluation order, and
//! records the same nodes the runtime tracer records when it executes the function:
//! inputs, data, then operations. Plain `f64` subexpressions (literals, constants such as
//! `Self::EPS`) are not nodes; literals fold at expansion time, other expressions stay
//! symbolic and are evaluated by the compiler in the generated code.
//!
//! Loops are not unrolled: a `for` loop becomes a [`Op::Loop`] marker, its body (traced
//! once, with the loop variable symbolic), an [`Op::End`], and a carried value
//! ([`Op::Phi`], [`Op::Next`], [`Op::Out`]) for every outer variable the body assigns.
//! Arrays written with a loop-dependent index are local arrays (`let mut a = [x; 8]`) or
//! array parameters, read and written through [`Op::Load`], [`Op::Store`], [`Op::Param`].
//! Indices the macro cannot evaluate (`MATURITIES[i]`, `interval(h)`) and plain `f64`
//! expressions of loop variables (`i as f64`, `weight(h)`) stay Rust source
//! ([`Opaque`]). A `let` of one becomes a local of the generated code, computed once where it
//! is defined and read by name: [`Op::LetInt`] for an integer (`let k: usize = ..;`),
//! [`Op::LetNum`] for an `f64` that reads a loop variable (`let w = weight(h);`), and
//! [`Op::LetConst`], of the type Rust infers, for constants only (`let n = LAST;`). A plain
//! `f64` that reads only locals of constants is still a constant.
//!
//! It also returns the function body rewritten so it compiles for any `N: AdNum`
//! (`2 * x` becomes `x.rmul(2f64)`, `x.maximum(0.0)` becomes `x.maximum(N::from(0.0))`),
//! which is the eager / tracer version of the kernel.

use proc_macro2::{Ident, Literal, Span, TokenStream, TokenTree};
use quote::{ToTokens, format_ident, quote, quote_spanned};
use raads_codegen::{Graph, Int, NONE, Op, Opaque, f64_literal};
use std::collections::{HashMap, HashSet};
use syn::spanned::Spanned;
use syn::{BinOp, Error, Expr, Lit, Pat, Result, Stmt, UnOp};

/// An `f64` that is not a graph node.
#[derive(Clone, Debug)]
pub enum F {
    Num(f64),
    /// Rust source for the generated code and the loop variables and locals it reads. One
    /// that reads loop variables, or locals computed from them, becomes an [`Op::Num`] node
    /// where a number is needed, not a constant.
    Expr(TokenStream, Vec<String>),
}

impl F {
    /// Rust source of the value, for the emitter.
    pub fn source(&self) -> String {
        match self {
            F::Num(k) => f64_literal(*k),
            F::Expr(t, _) => format!("({t})"),
        }
    }
    /// Loop variables and locals the value reads.
    fn vars(&self) -> &[String] {
        match self {
            F::Num(_) => &[],
            F::Expr(_, v) => v,
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
    /// A local array (the node of its [`Op::Array`]).
    Mem(usize),
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
    /// Array parameters still bound to their name: (input or data number of element 0,
    /// length, data).
    params: HashMap<String, (usize, usize, bool)>,
    /// Loop variables in scope, innermost last.
    ints: Vec<String>,
    /// Current value of local array elements read or written since the last loop boundary:
    /// (array, index source) → node. Reading an element twice gives the same node, as it
    /// does in the eager function.
    elems: HashMap<(usize, String), usize>,
    /// Variables the innermost loop carries to its next iteration.
    carried: Vec<String>,
    /// Integer locals (`let k: usize = ..;`): a literal, or the variable of their
    /// [`Op::LetInt`].
    ilocals: HashMap<String, Int>,
    /// Variables of [`Op::LetNum`]s, which are `f64`s (other variables are `usize`).
    num_lets: HashSet<String>,
    /// Variables of [`Op::LetInt`]s that read no loop variable, directly or through another
    /// let: plain values that read only these are constants.
    const_lets: HashSet<String>,
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
        let mut b = Builder {
            g: Graph::default(),
            consts: Vec::new(),
            env: HashMap::new(),
            params: HashMap::new(),
            ints: Vec::new(),
            elems: HashMap::new(),
            carried: Vec::new(),
            ilocals: HashMap::new(),
            num_lets: HashSet::new(),
            const_lets: HashSet::new(),
            n: n.clone(),
            krate: krate.clone(),
        };
        for data in [false, true] {
            let mut k = 0;
            for p in params.iter().filter(|p| p.data == data) {
                if let Shape::Array(len) = p.shape {
                    b.params.insert(p.name.clone(), (k, len, data));
                }
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

    /// `vars` include a loop variable, or a let computed from one.
    fn varies(&self, vars: &[String]) -> bool {
        vars.iter().any(|v| !self.const_lets.contains(v))
    }

    /// `v` is the value of an [`Op::LetNum`].
    fn num_let(&self, v: &Val) -> bool {
        matches!(v, Val::F(F::Expr(_, vars)) if vars.iter().any(|x| self.num_lets.contains(x)))
    }

    /// `v` is a plain `f64`: a number or the value of an [`Op::LetNum`] (an
    /// [`Op::LetConst`] may have any type).
    fn f64_local(&self, v: &Val) -> bool {
        matches!(v, Val::F(F::Num(_))) || self.num_let(v)
    }

    fn konst(&mut self, f: &F) -> usize {
        let op = match self.varies(f.vars()) {
            false => Op::Const(f.source()),
            true => Op::Num(Opaque { src: f.source(), vars: f.vars().to_vec(), eval: None }),
        };
        let i = self.push(op, NONE, NONE, NONE, NONE);
        self.consts[i] = Some(f.clone());
        i
    }

    fn node(&mut self, v: &Val, span: Span) -> Result<usize> {
        match v {
            Val::Node(i) => Ok(*i),
            Val::F(f) => Ok(self.konst(f)),
            Val::Arr(_) | Val::Mem(_) => err(span, "expected a number, found an array"),
        }
    }

    /// Bind `name` (a `let`, with the type `ty` the user wrote, or an assignment); it no
    /// longer names an array parameter.
    fn set(&mut self, name: String, v: Val, span: Span, ty: Option<&syn::Type>) -> Result<()> {
        if self.ints.contains(&name) {
            return err(span, &format!("`{name}` shadows a loop variable"));
        }
        let v = self.local(v, ty);
        self.params.remove(&name);
        self.ilocals.remove(&name);
        self.env.insert(name, v);
        Ok(())
    }

    /// `v` with each plain expression replaced by a local of the generated code, computed
    /// here, where the value is defined: not pasted where it is used. One of constants only
    /// keeps the type `ty` or the one Rust infers for it (`let n = LAST;` may be a `usize`).
    fn local(&mut self, v: Val, ty: Option<&syn::Type>) -> Val {
        match v {
            Val::F(F::Expr(t, vars)) if !self.varies(&vars) => {
                let ty = ty.map(|t| t.to_token_stream().to_string());
                let j = self.push(Op::LetConst { src: t.to_string(), ty }, NONE, NONE, NONE, NONE);
                Val::F(F::Expr(format_ident!("p{j}").into_token_stream(), vec![]))
            }
            Val::F(F::Expr(t, vars)) => {
                let j = self.push(Op::LetNum(Opaque { src: t.to_string(), vars, eval: None }), NONE, NONE, NONE, NONE);
                let name = format!("p{j}");
                self.num_lets.insert(name.clone());
                Val::F(F::Expr(name.parse().unwrap(), vec![name]))
            }
            Val::Arr(vs) => Val::Arr(vs.into_iter().map(|v| self.local(v, None)).collect()),
            v => v,
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
        let Some((Stmt::Expr(tail, None), stmts)) = block.stmts.split_last().filter(|(t, _)| !matches!(t, Stmt::Expr(Expr::ForLoop(_), _)))
        else {
            return err(block.span(), "a kernel must end with its result value");
        };
        let mut out = self.stmts(stmts)?;
        let (v, t) = self.expr(tail)?;
        out.extend(t);
        Ok((v, out))
    }

    /// Statements without a result; the rewritten source.
    fn stmts(&mut self, stmts: &[Stmt]) -> Result<TokenStream> {
        let mut out = TokenStream::new();
        for st in stmts {
            match st {
                Stmt::Local(l) if usize_name(&l.pat).is_some() => {
                    let id = usize_name(&l.pat).unwrap();
                    let name = id.to_string();
                    let Some(init) = l.init.as_ref().filter(|i| i.diverge.is_none()) else {
                        return err(l.span(), "`let` needs a value");
                    };
                    if self.ints.contains(&name) {
                        return err(id.span(), &format!("`{name}` shadows a loop variable"));
                    }
                    if self.carried.contains(&name) {
                        return err(
                            id.span(),
                            &format!("`{name}` is carried to the next iteration: a `let` here would hide it, use another name"),
                        );
                    }
                    let k = match self.int(&init.expr)? {
                        Int::Lit(k) => Int::Lit(k),
                        k => {
                            let fixed = !self.varies(&k.vars());
                            let var = format!("p{}", self.push(Op::LetInt(k), NONE, NONE, NONE, NONE));
                            if fixed {
                                self.const_lets.insert(var.clone());
                            }
                            Int::Var(var)
                        }
                    };
                    self.env.remove(&name);
                    self.params.remove(&name);
                    self.ilocals.insert(name, k);
                    out.extend(l.to_token_stream());
                }
                Stmt::Local(l) => {
                    let init = l.init.as_ref().ok_or_else(|| Error::new(l.span(), "`let` needs a value"))?;
                    if init.diverge.is_some() {
                        return err(l.span(), "`let ... else` is not supported in a kernel");
                    }
                    let (v, e) = self.expr(&init.expr)?;
                    if has_mem(&v) && !matches!(&*init.expr, Expr::Repeat(_)) {
                        return err(init.expr.span(), "copying a local array is not supported: index it instead");
                    }
                    self.bind(&l.pat, v, None)?;
                    // an `f64` local is an `f64` in the rewritten source too, so a type error
                    // points at the `let`
                    let pat = &l.pat;
                    let ty = match pat {
                        Pat::Ident(p) if self.env.get(&p.ident.to_string()).is_some_and(|v| self.num_let(v)) => {
                            quote_spanned!(p.span()=> : f64)
                        }
                        _ => quote!(),
                    };
                    out.extend(quote_spanned!(l.span()=> let #pat #ty = #e;));
                }
                Stmt::Expr(Expr::Assign(a), _) if matches!(&*a.left, Expr::Index(_)) => {
                    let Expr::Index(ix) = &*a.left else { unreachable!() };
                    let name = ident_of(&ix.expr)?;
                    let Some(&Val::Mem(m)) = self.env.get(&name) else {
                        return err(ix.expr.span(), "only local arrays (`let mut a = [x; 8];`) can be assigned by element");
                    };
                    let idx = self.int(&ix.index)?;
                    let (v, e) = self.expr(&a.right)?;
                    self.check_index(&idx, m, ix.index.span())?;
                    let val = self.node(&v, a.right.span())?;
                    // any element of `m` may be the one written
                    self.elems.retain(|&(a, _), _| a != m);
                    self.elems.insert((m, idx.to_string()), val);
                    self.push(Op::Store(idx), m, val, NONE, NONE);
                    let (l, e) = (&a.left, self.as_n(&v, e));
                    out.extend(quote!(#l = #e;));
                }
                Stmt::Expr(Expr::Assign(a), _) => {
                    let name = ident_of(&a.left)?;
                    if self.ilocals.contains_key(&name) {
                        return err(a.left.span(), &format!("`{name}` is a `usize` local: kernels do not reassign integers"));
                    }
                    match self.env.get(&name) {
                        None => return err(a.left.span(), &format!("unknown variable `{name}`")),
                        Some(Val::Mem(_)) => return err(a.left.span(), "a local array is assigned by element: `a[i] = x;`"),
                        _ => {}
                    }
                    let (v, e) = self.expr(&a.right)?;
                    if has_mem(&v) {
                        return err(a.right.span(), "copying a local array is not supported: index it instead");
                    }
                    self.set(name, v, a.left.span(), None)?;
                    let l = &a.left;
                    out.extend(quote!(#l = #e;));
                }
                Stmt::Expr(Expr::ForLoop(f), _) => out.extend(self.for_loop(f)?),
                _ => return err(st.span(), "kernel statements are `let` bindings, assignments, `for` loops and a final value"),
            }
        }
        Ok(out)
    }

    /// `for i in lo..hi { .. }`: the body is traced once. Outer variables it assigns are
    /// carried from one iteration to the next.
    fn for_loop(&mut self, f: &syn::ExprForLoop) -> Result<TokenStream> {
        let Pat::Ident(pi) = &*f.pat else { return err(f.pat.span(), "a loop variable is a plain name") };
        let var = pi.ident.to_string();
        if crate::is_reserved(&var) {
            return err(pi.span(), &format!("loop variable name `{var}` is reserved by the generated code ({})", crate::RESERVED));
        }
        let Expr::Range(r) = &*f.expr else { return err(f.expr.span(), "a kernel loop runs over a range `lo..hi`") };
        let (Some(start), Some(end), syn::RangeLimits::HalfOpen(_)) = (&r.start, &r.end, &r.limits) else {
            return err(r.span(), "a kernel loop runs over a range `lo..hi`");
        };
        let (lo, hi) = (self.int(start)?, self.int(end)?);
        let loop_vars_only = |b: &Int| !b.has_opaque() && b.vars().iter().all(|v| self.ints.contains(v));
        if !loop_vars_only(&lo) || !loop_vars_only(&hi) {
            return err(
                r.span(),
                "loop bounds are integer literals, loop variables and `+ - * / %` of them (the macro sizes the tape from them)",
            );
        }
        let mut carried = Vec::new();
        assigned(&f.body.stmts, &mut HashSet::new(), &mut carried);
        carried.retain(|name| self.env.contains_key(name));
        // initial values, before the loop
        let mut inits = Vec::new();
        for name in &carried {
            match &self.env[name] {
                Val::Node(v) => inits.push(*v),
                Val::F(_) => return err(f.span(), &format!("`{name}` is a plain f64 changed in a loop: start it as `N::from(..)`")),
                _ => return err(f.span(), &format!("`{name}` is an array: assign a local array by element in a loop")),
            }
        }
        let (env, params, ilocals) = (self.env.clone(), self.params.clone(), self.ilocals.clone());
        self.elems.clear();
        self.push(Op::Loop(var.clone(), lo, hi), NONE, NONE, NONE, NONE);
        let phis: Vec<usize> = inits.iter().map(|&init| self.push(Op::Phi, init, NONE, NONE, NONE)).collect();
        for (name, &p) in carried.iter().zip(&phis) {
            self.env.insert(name.clone(), Val::Node(p));
        }
        self.ints.push(var);
        let outer = std::mem::replace(&mut self.carried, carried.clone());
        let body = self.stmts(&f.body.stmts);
        self.carried = outer;
        self.ints.pop();
        let body = body?;
        for (name, &p) in carried.iter().zip(&phis) {
            let v = self.env[name].clone();
            let next = self.node(&v, f.span())?;
            self.push(Op::Next, p, next, NONE, NONE);
        }
        self.elems.clear();
        self.push(Op::End, NONE, NONE, NONE, NONE);
        (self.env, self.params, self.ilocals) = (env, params, ilocals);
        for (name, &p) in carried.iter().zip(&phis) {
            let out = self.push(Op::Out, p, NONE, NONE, NONE);
            self.env.insert(name.clone(), Val::Node(out));
        }
        let (pat, range, attrs) = (&f.pat, &f.expr, &f.attrs);
        Ok(quote!(#(#attrs)* for #pat in #range { #body }))
    }

    /// An integer expression: literals, loop variables, integer locals and `+ - * / %` of
    /// them; anything else is plain Rust.
    fn int(&self, e: &Expr) -> Result<Int> {
        let name = match e {
            Expr::Path(p) => p.path.get_ident().map(|id| id.to_string()),
            _ => None,
        };
        match e {
            Expr::Lit(syn::ExprLit { lit: Lit::Int(k), .. }) => Ok(Int::Lit(k.base10_parse()?)),
            _ if name.as_ref().is_some_and(|n| self.ints.contains(n)) => Ok(Int::Var(name.unwrap())),
            _ if name.as_ref().is_some_and(|n| self.ilocals.contains_key(n)) => Ok(self.ilocals[&name.unwrap()].clone()),
            _ if name.as_ref().is_some_and(|n| self.env.get(n).is_some_and(|v| self.f64_local(v))) => {
                let n = name.unwrap();
                err(e.span(), &format!("`{n}` is an `f64`: an integer local is declared with its type, `let {n}: usize = ..;`"))
            }
            Expr::Paren(p) => self.int(&p.expr),
            Expr::Group(g) => self.int(&g.expr),
            Expr::Binary(b) if matches!(b.op, BinOp::Add(_) | BinOp::Sub(_) | BinOp::Mul(_) | BinOp::Div(_) | BinOp::Rem(_)) => {
                let op = match b.op {
                    BinOp::Add(_) => '+',
                    BinOp::Sub(_) => '-',
                    BinOp::Mul(_) => '*',
                    BinOp::Div(_) => '/',
                    _ => '%',
                };
                Ok(Int::Bin(Box::new(self.int(&b.left)?), op, Box::new(self.int(&b.right)?)))
            }
            _ => {
                let (t, vars) = self.plain(e)?;
                Ok(Int::Opaque(Opaque { src: t.to_string(), vars, eval: None }))
            }
        }
    }

    /// A plain Rust expression (`MATURITIES[i]`, `weight(h)`), which the macro does not
    /// interpret: its source for the generated code, with integer and plain `f64` locals
    /// replaced by their variables, and the variables it reads.
    fn plain(&self, e: &impl ToTokens) -> Result<(TokenStream, Vec<String>)> {
        let mut vars = Vec::new();
        let t = self.plain_tokens(e.to_token_stream(), &mut vars)?;
        Ok((t, vars))
    }

    fn plain_tokens(&self, ts: TokenStream, vars: &mut Vec<String>) -> Result<TokenStream> {
        let trees: Vec<TokenTree> = ts.into_iter().collect();
        let mut out = TokenStream::new();
        for (k, tt) in trees.iter().enumerate() {
            match tt {
                TokenTree::Group(g) => {
                    let mut inner = proc_macro2::Group::new(g.delimiter(), self.plain_tokens(g.stream(), vars)?);
                    inner.set_span(g.span());
                    out.extend([TokenTree::Group(inner)]);
                    continue;
                }
                TokenTree::Ident(id) if !in_path(&trees, k) => {
                    let name = id.to_string();
                    if self.ints.contains(&name) {
                        add_vars(vars, std::slice::from_ref(&name));
                    } else if let Some(i) = self.ilocals.get(&name) {
                        add_vars(vars, &i.vars());
                        let t: TokenStream = i.to_string().parse().unwrap();
                        out.extend(quote!((#t)));
                        continue;
                    } else if let Some(v) = self.env.get(&name) {
                        let Val::F(f) = v else {
                            return err(
                                id.span(),
                                &format!(
                                    "`{name}` is a kernel value: plain Rust expressions read loop variables, integers and `f64` constants"
                                ),
                            );
                        };
                        add_vars(vars, f.vars());
                        let t = f.tokens();
                        out.extend(quote!((#t)));
                        continue;
                    }
                }
                _ => {}
            }
            out.extend([tt.clone()]);
        }
        Ok(out)
    }

    /// A constant index is within the local array `m`.
    fn check_index(&self, idx: &Int, m: usize, span: Span) -> Result<()> {
        let Op::Array(len) = self.g.nodes[m].op else { unreachable!() };
        match idx.constant() {
            Some(k) if k >= len => err(span, "index out of bounds"),
            _ => Ok(()),
        }
    }

    /// `ty`: the type the user wrote for `pat`.
    fn bind(&mut self, pat: &Pat, v: Val, ty: Option<&syn::Type>) -> Result<()> {
        match (pat, v) {
            (Pat::Ident(p), _) if self.carried.contains(&p.ident.to_string()) => {
                err(p.span(), &format!("`{}` is carried to the next iteration: a `let` here would hide it, use another name", p.ident))
            }
            (Pat::Ident(p), v) => self.set(p.ident.to_string(), v, p.span(), ty),
            (Pat::Type(t), v) => self.bind(&t.pat, v, Some(&t.ty)),
            (Pat::Wild(_), _) => Ok(()),
            (Pat::Tuple(t), Val::Arr(vs)) if t.elems.len() == vs.len() => {
                t.elems.iter().zip(vs).try_for_each(|(p, v)| self.bind(p, v, None))
            }
            (Pat::Slice(s), Val::Arr(vs)) if s.elems.len() == vs.len() => {
                s.elems.iter().zip(vs).try_for_each(|(p, v)| self.bind(p, v, None))
            }
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
                    if self.ints.contains(&name) {
                        return err(sp, &format!("`{name}` is a loop variable (`usize`): write `{name} as f64` for its value"));
                    }
                    if self.ilocals.contains_key(&name) {
                        return err(sp, &format!("`{name}` is a `usize`: write `{name} as f64` for its value"));
                    }
                    if let Some(v) = self.env.get(&name) {
                        return Ok((v.clone(), e.to_token_stream()));
                    }
                    if !is_constant_name(&name) {
                        return err(sp, &format!("unknown variable `{name}` (constants are SCREAMING_CASE or paths)"));
                    }
                }
                Ok((Val::F(F::Expr(e.to_token_stream(), vec![])), e.to_token_stream()))
            }
            Expr::Index(ix) => {
                let idx = self.int(&ix.index)?;
                let i = &ix.index;
                let param = match &*ix.expr {
                    Expr::Path(p) => p.path.get_ident().and_then(|id| self.params.get(&id.to_string())).copied(),
                    _ => None,
                };
                if let (Some((first, len, data)), None) = (param, idx.constant()) {
                    let a = &ix.expr;
                    let v = self.push(Op::Param { first, len, data, idx }, NONE, NONE, NONE, NONE);
                    return Ok((Val::Node(v), quote!(#a[#i])));
                }
                let (v, t) = self.expr(&ix.expr)?;
                let v = match (v, idx.constant()) {
                    (Val::Arr(vs), Some(k)) => vs.get(k).cloned().ok_or_else(|| Error::new(ix.index.span(), "index out of bounds"))?,
                    (Val::Arr(_), None) => {
                        return err(ix.index.span(), "a computed index needs an array parameter or a local array (`let mut a = [x; 8];`)");
                    }
                    (Val::Mem(m), _) => {
                        self.check_index(&idx, m, ix.index.span())?;
                        let key = (m, idx.to_string());
                        match self.elems.get(&key) {
                            Some(&v) => Val::Node(v),
                            None => {
                                let v = self.push(Op::Load(idx), m, NONE, NONE, NONE);
                                self.elems.insert(key, v);
                                Val::Node(v)
                            }
                        }
                    }
                    (Val::F(_), _) => {
                        // a plain Rust array (`SWAPRATES[i]`)
                        let (t, vars) = self.plain(e)?;
                        return Ok((Val::F(F::Expr(t, vars)), e.to_token_stream()));
                    }
                    _ => return err(sp, "indexing a value that is not an array"),
                };
                Ok((v, quote!(#t[#i])))
            }
            Expr::Repeat(r) => {
                let Some(len) = self.int(&r.len)?.constant() else { return err(r.len.span(), "local array lengths are integer literals") };
                let (v, t) = self.expr(&r.expr)?;
                let fill = self.node(&v, r.expr.span())?;
                let m = self.push(Op::Array(len), fill, NONE, NONE, NONE);
                let (t, l) = (self.as_n(&v, t), &r.len);
                Ok((Val::Mem(m), quote!([#t; #l])))
            }
            Expr::Cast(c) if matches!(&*c.ty, syn::Type::Path(p) if p.path.is_ident("f64")) => {
                let idx = match self.int(&c.expr) {
                    Ok(idx) => idx,
                    // `w as f64` of a plain `f64` (`w = weight(i)`) is `w`
                    Err(e) => match self.expr(&c.expr) {
                        Ok((Val::F(f), _)) => return Ok((Val::F(f), c.to_token_stream())),
                        _ => return Err(e),
                    },
                };
                if let Some(k) = idx.constant() {
                    return Ok((Val::F(F::Num(k as f64)), e.to_token_stream()));
                }
                let t: TokenStream = format!("{idx} as f64").parse().unwrap();
                Ok((Val::F(F::Expr(t, idx.vars())), e.to_token_stream()))
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
                        Val::F(F::Expr(quote!(-#t), f.vars().to_vec()))
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
                        let mut vars = a.vars().to_vec();
                        add_vars(&mut vars, c.vars());
                        F::Expr(quote!(#x #sym #y), vars)
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
                        F::Expr(t, v) => F::Expr(quote!(1.0_f64 / (#t)), v.clone()),
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
            let mut vars = f.vars().to_vec();
            for a in &args {
                if let Val::F(x) = a {
                    add_vars(&mut vars, x.vars());
                }
            }
            return Ok((Val::F(F::Expr(quote!(#ft.#meth(#(#fa),*)), vars)), src));
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
                if self.varies(k.vars()) {
                    return err(m.args[0].span(), "the exponent of `powf` cannot depend on a loop variable");
                }
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
                    F::Expr(t, v) => F::Expr(quote!(1.0_f64 / (#t)), v.clone()),
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
        if last != "select_gt" && last != "from" {
            // a plain Rust function (`weight(h)`)
            let (t, vars) = self.plain(c)?;
            return Ok((Val::F(F::Expr(t, vars)), c.to_token_stream()));
        }
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
            _ => err(sp, "`from` takes one argument, `select_gt` four"),
        }
    }

    /// Graph outputs from the result value, checked against the declared shape.
    pub fn outputs(&mut self, v: &Val, shape: Shape, span: Span) -> Result<()> {
        let outs = match (shape, v) {
            (Shape::Scalar, v) => vec![self.node(v, span)?],
            (Shape::Array(n), Val::Arr(vs)) if vs.len() == n => vs.iter().map(|x| self.node(x, span)).collect::<Result<_>>()?,
            (Shape::Array(n), &Val::Mem(m)) if self.g.nodes[m].op == Op::Array(n) => {
                (0..n).map(|k| self.push(Op::Load(Int::Lit(k)), m, NONE, NONE, NONE)).collect()
            }
            _ => return err(span, "result does not match the declared return type"),
        };
        self.g.outs = outs;
        Ok(())
    }

    /// Code that rebuilds the graph at run time (constants evaluated by the compiler),
    /// for comparison with the tracer.
    pub fn graph_code(&self) -> TokenStream {
        let krate = &self.krate;
        // constants read the lets of constants by name
        let lets = self.const_defs(None);
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
                Op::Loop(var, lo, hi) => {
                    let (lo, hi) = (self.int_code(lo), self.int_code(hi));
                    quote!(Loop(#var.to_string(), #lo, #hi))
                }
                Op::End => quote!(End),
                Op::Phi => quote!(Phi),
                Op::Next => quote!(Next),
                Op::Out => quote!(Out),
                Op::Array(len) => quote!(Array(#len)),
                Op::Load(idx) => {
                    let idx = self.int_code(idx);
                    quote!(Load(#idx))
                }
                Op::Store(idx) => {
                    let idx = self.int_code(idx);
                    quote!(Store(#idx))
                }
                Op::Param { first, len, data, idx } => {
                    let idx = self.int_code(idx);
                    quote!(Param { first: #first, len: #len, data: #data, idx: #idx })
                }
                Op::Num(x) => {
                    let x = self.opaque_code(x, quote!(f64));
                    quote!(Num(#x))
                }
                Op::LetInt(idx) => {
                    let idx = self.int_code(idx);
                    quote!(LetInt(#idx))
                }
                Op::LetNum(x) => {
                    let x = self.opaque_code(x, quote!(f64));
                    quote!(LetNum(#x))
                }
                Op::LetConst { src, ty } => {
                    let ty = match ty {
                        Some(t) => quote!(Some(#t.to_string())),
                        None => quote!(None),
                    };
                    quote!(LetConst { src: #src.to_string(), ty: #ty })
                }
            };
            quote!(g.push(#krate::__private::codegen::Op::#op, #a, #b, #c, #d);)
        });
        let outs = &self.g.outs;
        quote! {
            #lets
            let mut g = #krate::__private::codegen::Graph::default();
            #(#nodes)*
            g.outs = vec![#(#outs),*];
            g
        }
    }

    /// Code that builds `i`.
    fn int_code(&self, i: &Int) -> TokenStream {
        let krate = &self.krate;
        match i {
            Int::Lit(k) => quote!(#krate::__private::codegen::Int::Lit(#k)),
            Int::Var(v) => quote!(#krate::__private::codegen::Int::Var(#v.to_string())),
            Int::Bin(a, op, b) => {
                let (a, b) = (self.int_code(a), self.int_code(b));
                quote!(#krate::__private::codegen::Int::Bin(Box::new(#a), #op, Box::new(#b)))
            }
            Int::Opaque(x) => {
                let x = self.opaque_code(x, quote!(usize));
                quote!(#krate::__private::codegen::Int::Opaque(#x))
            }
        }
    }

    /// `let`s of the locals of constants ([`Op::LetConst`], constant [`Op::LetInt`]s), in order:
    /// all of them, or those the Rust source `src` reads.
    fn const_defs(&self, src: Option<&str>) -> TokenStream {
        let defs: Vec<_> = (self.g.nodes.iter().enumerate())
            .filter_map(|(i, n)| match &n.op {
                Op::LetInt(k) if self.const_lets.contains(&format!("p{i}")) => Some((i, Some("usize".to_string()), k.to_string())),
                Op::LetConst { src, ty } => Some((i, ty.clone(), src.clone())),
                _ => None,
            })
            .collect();
        // a def reads only earlier ones
        let mut need = src.map(idents);
        for (i, _, s) in defs.iter().rev() {
            if let Some(need) = need.as_mut().filter(|n| n.contains(&format!("p{i}"))) {
                need.extend(idents(s));
            }
        }
        let defs = defs.iter().filter(|(i, ..)| need.as_ref().is_none_or(|n| n.contains(&format!("p{i}")))).map(|(i, ty, s)| {
            let id = format_ident!("p{i}");
            let ty: TokenStream = ty.as_ref().map(|t| format!(": {t}")).unwrap_or_default().parse().unwrap();
            let t: TokenStream = s.parse().unwrap();
            quote!(#[allow(unused_variables)] let #id #ty = #t;)
        });
        quote!(#(#defs)*)
    }

    /// Code that builds `x`, with a function that evaluates it from its variables (and computes
    /// the locals of constants it may read).
    fn opaque_code<T>(&self, x: &Opaque<T>, ty: TokenStream) -> TokenStream {
        let krate = &self.krate;
        let (src, vars) = (&x.src, &x.vars);
        let defs = self.const_defs(Some(src));
        let t: TokenStream = src.parse().unwrap();
        let binds = vars.iter().enumerate().map(|(k, v)| {
            let id = Ident::new(v, Span::call_site());
            match self.num_lets.contains(v) {
                true => quote!(#[allow(unused_variables)] let #id: f64 = __raads_vars[#k];),
                false => quote!(#[allow(unused_variables)] let #id: usize = __raads_vars[#k] as usize;),
            }
        });
        quote! {
            #krate::__private::codegen::Opaque {
                src: #src.to_string(),
                vars: vec![#(#vars.to_string()),*],
                eval: Some(|__raads_vars: &[f64]| -> #ty { #defs #(#binds)* #t }),
            }
        }
    }
}

/// The identifiers in the Rust source `src`.
fn idents(src: &str) -> HashSet<String> {
    fn walk(ts: TokenStream, out: &mut HashSet<String>) {
        for tt in ts {
            match tt {
                TokenTree::Ident(id) => {
                    out.insert(id.to_string());
                }
                TokenTree::Group(g) => walk(g.stream(), out),
                _ => {}
            }
        }
    }
    let mut out = HashSet::new();
    walk(src.parse().unwrap(), &mut out);
    out
}

/// `v` is or contains a local array.
fn has_mem(v: &Val) -> bool {
    match v {
        Val::Mem(_) => true,
        Val::Arr(vs) => vs.iter().any(has_mem),
        _ => false,
    }
}

/// Add the variables `new` to `vars`.
fn add_vars(vars: &mut Vec<String>, new: &[String]) {
    for v in new {
        if !vars.contains(v) {
            vars.push(v.clone());
        }
    }
}

/// Token `k` is a field, method or path segment (after `.` or `::`) or a path prefix
/// (before `::`), not a variable.
fn in_path(trees: &[TokenTree], k: usize) -> bool {
    let punct = |j: Option<usize>, c: char| matches!(j.and_then(|j| trees.get(j)), Some(TokenTree::Punct(p)) if p.as_char() == c);
    punct(k.checked_sub(1), '.') || punct(k.checked_sub(1), ':') || (punct(Some(k + 1), ':') && punct(Some(k + 2), ':'))
}

/// The name in `let name: usize = ..`.
fn usize_name(p: &Pat) -> Option<&Ident> {
    match p {
        Pat::Type(t) if matches!(&*t.ty, syn::Type::Path(ty) if ty.path.is_ident("usize")) => match &*t.pat {
            Pat::Ident(id) => Some(&id.ident),
            _ => None,
        },
        _ => None,
    }
}

/// Outer variables the statements assign (not declared by them first), in order.
fn assigned(stmts: &[Stmt], locals: &mut HashSet<String>, out: &mut Vec<String>) {
    fn names(p: &Pat, locals: &mut HashSet<String>) {
        match p {
            Pat::Ident(i) => {
                locals.insert(i.ident.to_string());
            }
            Pat::Type(t) => names(&t.pat, locals),
            Pat::Tuple(t) => t.elems.iter().for_each(|p| names(p, locals)),
            Pat::Slice(s) => s.elems.iter().for_each(|p| names(p, locals)),
            _ => {}
        }
    }
    for st in stmts {
        match st {
            Stmt::Local(l) => names(&l.pat, locals),
            Stmt::Expr(Expr::Assign(a), _) => {
                if let Ok(name) = ident_of(&a.left)
                    && !locals.contains(&name)
                    && !out.contains(&name)
                {
                    out.push(name);
                }
            }
            Stmt::Expr(Expr::ForLoop(f), _) => {
                let mut inner = locals.clone();
                names(&f.pat, &mut inner);
                assigned(&f.body.stmts, &mut inner, out);
            }
            _ => {}
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
