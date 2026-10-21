//! `#[kernels(engine)]` and `check_kernels!`: engine-agnostic model compilation.
//!
//! 1. `#[kernels(path)]` compiles every function of an inherent `impl` block like `#[kernel]`
//!    (the generated functions are named `<kernel>_eval`, `_vjp`, `_fwd`, `_bwd`) and expands to
//!    the block plus `path::__raads_model! { [raads] [generics] [Self type] [where predicates]
//!    [const names] [kernels] }`, where each kernel is `{ (original signature;) NRES [eval] [vjp]
//!    [fwd] [bwd] }` with the user's tokens and spans.
//! 2. The engine's `__raads_model!` (a `macro_rules!`) forwards that, minus the `raads` path, to
//!    `raads::__private::check_kernels! { callback; spec { fn ..; const ..; } ... }`.
//! 3. `check_kernels!` checks the kernels against the spec (errors at the offending tokens)
//!    and expands to `callback! { @impl [generics] [Self type] [where predicates] [consts]
//!    [dims] [kernels] }`: dims are the lengths of the spec's named dimensions in order of
//!    first appearance, kernels are `name NRES [eval] [vjp] [fwd] [bwd]` in spec order, consts
//!    forward the spec's consts the block defines (`const K: T = <Self>::K;`), so they stay
//!    inherent and usable as `Self::K` in kernels without the trait in scope.

use crate::front::Shape;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{
    Error, FnArg, GenericParam, Ident, ImplItem, ImplItemFn, ItemFn, ItemImpl, Pat, Result, ReturnType, TraitItem, Type, braced, bracketed,
};

fn combine(errors: &mut Option<Error>, e: Error) {
    match errors {
        Some(all) => all.combine(e),
        None => *errors = Some(e),
    }
}

/// `#[kernels(engine)] impl Model { ... }`.
pub fn kernels(engine: syn::Path, mut imp: ItemImpl) -> Result<TokenStream> {
    if imp.trait_.is_some() {
        return Err(Error::new(imp.span(), "put `#[kernels]` on an inherent `impl Model { ... }` block"));
    }
    // the generated functions are generic over `R`
    if let Some(p) = imp.generics.params.iter().find(|p| matches!(p, GenericParam::Type(t) if t.ident == "R")) {
        return Err(Error::new(p.span(), "the generic parameter name `R` is reserved by the generated code"));
    }
    let krate = crate::krate();
    let mut kernels = Vec::new();
    let mut errors = None;
    for item in imp.items.iter_mut() {
        let ImplItem::Fn(f) = item else { continue };
        let sig = f.sig.clone();
        let n = |k: &str| format!("{}_{k}", sig.ident);
        match crate::compile(to_item_fn(f), "", [&n("eval"), &n("vjp"), &n("fwd"), &n("bwd")], &krate) {
            Ok(c) => {
                f.sig = c.source.sig;
                f.block = *c.source.block;
                let (nres, fns) = (c.nres, [c.eval, c.vjp, c.fwd, c.bwd]);
                kernels.push(quote!({ (#sig;) #nres #([#[doc(hidden)] #fns])* }));
            }
            Err(e) => combine(&mut errors, e),
        }
    }
    if let Some(e) = errors {
        return Err(e);
    }
    let consts = imp.items.iter().filter_map(|it| match it {
        ImplItem::Const(k) => Some(&k.ident),
        _ => None,
    });
    let (params, ty) = (&imp.generics.params, &imp.self_ty);
    let wh = imp.generics.where_clause.as_ref().map(|w| &w.predicates);
    Ok(quote! {
        #imp
        #engine::__raads_model! { [#krate] [#params] [#ty] [#wh] [#(#consts)*] [#(#kernels)*] }
    })
}

fn to_item_fn(f: &ImplItemFn) -> ItemFn {
    ItemFn { attrs: f.attrs.clone(), vis: f.vis.clone(), sig: f.sig.clone(), block: Box::new(f.block.clone()) }
}

/// Expected shape of a kernel parameter or result: `N`, `[N; k]`, or `[N; DIM]` with a
/// dimension shared across the kernels.
#[derive(Clone)]
enum Dim {
    Scalar,
    Fixed(usize),
    Var(String),
}

impl Dim {
    fn parse(ty: &Type) -> Result<Dim> {
        match ty {
            Type::Array(a) => match &a.len {
                syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(k), .. }) => Ok(Dim::Fixed(k.base10_parse()?)),
                syn::Expr::Path(p) if p.path.get_ident().is_some() => Ok(Dim::Var(p.path.get_ident().unwrap().to_string())),
                e => Err(Error::new(e.span(), "spec array lengths are integer literals or dimension names")),
            },
            _ => Ok(Dim::Scalar),
        }
    }

    fn show(&self) -> String {
        match self {
            Dim::Scalar => "N".into(),
            Dim::Fixed(k) => format!("[N; {k}]"),
            Dim::Var(v) => format!("[N; {v}]"),
        }
    }
}

/// Expected signature of one kernel: parameters (name, shape, `#[data]`) and result.
struct Spec {
    name: String,
    params: Vec<(String, Dim, bool)>,
    ret: Dim,
}

impl Spec {
    fn signature(&self) -> String {
        let ps: Vec<String> =
            self.params.iter().map(|(n, d, data)| format!("{}{n}: {}", if *data { "#[data] " } else { "" }, d.show())).collect();
        format!("`{}({}) -> {}`", self.name, ps.join(", "), self.ret.show())
    }
}

/// A parameter: name, `#[data]`, type.
fn param(arg: &FnArg) -> Result<(String, bool, &Type)> {
    let FnArg::Typed(pt) = arg else { return Err(Error::new(arg.span(), "kernels take no `self`")) };
    let Pat::Ident(id) = &*pt.pat else { return Err(Error::new(pt.pat.span(), "parameters must be plain names")) };
    Ok((id.ident.to_string(), pt.attrs.iter().any(|a| a.path().is_ident("data")), &pt.ty))
}

/// A kernel of the user's block, as `#[kernels]` passed it.
struct Kernel {
    sig: syn::Signature,
    nres: syn::LitInt,
    fns: [TokenStream; 4],
}

/// One `[...]` group's contents.
fn group(input: ParseStream) -> Result<TokenStream> {
    let g;
    bracketed!(g in input);
    g.parse()
}

pub struct Input {
    callback: TokenStream,
    specs: Vec<Spec>,
    /// Spec consts: name and type.
    consts: Vec<(Ident, Type)>,
    generics: TokenStream,
    ty: Type,
    wh: TokenStream,
    defined: Vec<Ident>,
    kernels: Vec<Kernel>,
}

impl Parse for Input {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut callback = TokenStream::new();
        while !input.peek(syn::Token![;]) {
            callback.extend([input.parse::<proc_macro2::TokenTree>()?]);
        }
        input.parse::<syn::Token![;]>()?;
        let kw: Ident = input.parse()?;
        if kw != "spec" {
            return Err(Error::new(kw.span(), "expected `spec { .. }`"));
        }
        let body;
        braced!(body in input);
        let (mut specs, mut consts) = (Vec::new(), Vec::new());
        while !body.is_empty() {
            match body.parse::<TraitItem>()? {
                TraitItem::Fn(f) => {
                    let params = f
                        .sig
                        .inputs
                        .iter()
                        .map(|a| {
                            let (name, data, ty) = param(a)?;
                            Ok((name, Dim::parse(ty)?, data))
                        })
                        .collect::<Result<_>>()?;
                    let ret = match &f.sig.output {
                        ReturnType::Type(_, t) => Dim::parse(t)?,
                        ReturnType::Default => return Err(Error::new(f.sig.span(), "a spec kernel has a result")),
                    };
                    specs.push(Spec { name: f.sig.ident.to_string(), params, ret });
                }
                TraitItem::Const(k) => consts.push((k.ident, k.ty)),
                it => return Err(Error::new(it.span(), "a spec lists kernels (`fn`) and consts")),
            }
        }
        let generics = group(input)?;
        let ty = {
            let g;
            bracketed!(g in input);
            g.parse()?
        };
        let wh = group(input)?;
        let defined = {
            let g;
            bracketed!(g in input);
            let mut v = Vec::new();
            while !g.is_empty() {
                v.push(g.parse()?);
            }
            v
        };
        let g;
        bracketed!(g in input);
        let mut kernels = Vec::new();
        while !g.is_empty() {
            let k;
            braced!(k in g);
            let s;
            syn::parenthesized!(s in k);
            let sig = s.parse::<syn::TraitItemFn>()?.sig;
            let nres = k.parse()?;
            let fns = [group(&k)?, group(&k)?, group(&k)?, group(&k)?];
            kernels.push(Kernel { sig, nres, fns });
        }
        Ok(Input { callback, specs, consts, generics, ty, wh, defined, kernels })
    }
}

/// A user kernel's parameters (shape, `#[data]`, type span) and result (shape, span).
struct Shapes {
    params: Vec<(Shape, bool, Span)>,
    ret: (Shape, Span),
}

fn shapes(sig: &syn::Signature) -> Result<Shapes> {
    let n = sig.generics.type_params().next().map(|t| t.ident.clone()).ok_or_else(|| Error::new(sig.span(), "missing `<N: AdNum>`"))?;
    let params = sig
        .inputs
        .iter()
        .map(|a| {
            let (_, data, ty) = param(a)?;
            Ok((crate::shape(ty, &n)?, data, ty.span()))
        })
        .collect::<Result<_>>()?;
    let ret = match &sig.output {
        ReturnType::Type(_, t) => (crate::shape(t, &n)?, t.span()),
        ReturnType::Default => return Err(Error::new(sig.span(), "a kernel returns `N` or `[N; LEN]`")),
    };
    Ok(Shapes { params, ret })
}

/// `check_kernels!`: see the module docs.
pub fn check(input: Input) -> Result<TokenStream> {
    let Input { callback, specs, consts, generics, ty, wh, defined, kernels } = input;
    let mut errors = None;
    let mut found = Vec::new();
    for spec in &specs {
        match kernels.iter().find(|k| k.sig.ident == spec.name) {
            Some(k) => found.push(k),
            None => combine(&mut errors, Error::new(ty.span(), format!("missing kernel {}", spec.signature()))),
        }
    }
    for k in &kernels {
        if !specs.iter().any(|s| k.sig.ident == s.name) {
            let names: Vec<String> = specs.iter().map(|s| format!("`{}`", s.name)).collect();
            let msg = format!("`{}` is not a kernel of this engine (expected {})", k.sig.ident, names.join(", "));
            combine(&mut errors, Error::new(k.sig.ident.span(), msg));
        }
    }
    if let Some(e) = errors {
        return Err(e);
    }

    // dimension -> (length, kernel that fixed it), in order of first appearance
    let mut dims: Vec<(String, usize, String)> = Vec::new();
    for (spec, k) in specs.iter().zip(&found) {
        let sig = spec.signature();
        if k.sig.inputs.len() != spec.params.len() {
            combine(&mut errors, Error::new(k.sig.paren_token.span.join(), format!("expected {sig}")));
            continue;
        }
        let s = shapes(&k.sig)?;
        let mut shape = |d: &Dim, s: Shape, span: Span| {
            let found = match (d, s) {
                (Dim::Scalar, Shape::Scalar) => return,
                (Dim::Fixed(k), Shape::Array(n)) if *k == n => return,
                (Dim::Var(v), Shape::Array(n)) => match dims.iter().find(|(w, ..)| w == v) {
                    None => return dims.push((v.clone(), n, spec.name.clone())),
                    Some((_, m, _)) if *m == n => return,
                    Some((_, m, from)) => format!("`[N; {m}]` ({v}, as in `{from}`)"),
                },
                _ => format!("`{}`", d.show()),
            };
            combine(&mut errors, Error::new(span, format!("expected {found} in {sig}")));
        };
        for ((_, d, _), p) in spec.params.iter().zip(&s.params) {
            shape(d, p.0, p.2);
        }
        shape(&spec.ret, s.ret.0, s.ret.1);
        for ((name, _, data), p) in spec.params.iter().zip(&s.params) {
            if p.1 != *data {
                let msg = if *data { "must be `#[data]`" } else { "must not be `#[data]`" };
                combine(&mut errors, Error::new(p.2, format!("`{name}` {msg} in {sig}")));
            }
        }
    }
    if let Some(e) = errors {
        return Err(e);
    }

    let consts = consts.iter().filter(|(id, _)| defined.contains(id)).map(|(id, t)| quote!(const #id: #t = <#ty>::#id;));
    let lens = dims.iter().map(|d| d.1);
    let ks = specs.iter().zip(&found).map(|(spec, k)| {
        let (name, nres, [eval, vjp, fwd, bwd]) = (format_ident!("{}", spec.name), &k.nres, &k.fns);
        quote!(#name #nres [#eval] [#vjp] [#fwd] [#bwd])
    });
    Ok(quote! {
        #callback! { @impl [#generics] [#ty] [#wh] [#(#consts)*] [#(#lens)*] [#(#ks)*] }
    })
}
