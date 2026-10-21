//! Attribute macros for `raads`: `#[kernel]` turns a function written against `N: AdNum`
//! into straight-line forward and reverse code; `#[kernels(engine)]` compiles the kernels of
//! an `impl` block for an engine. Re-exported by `raads`; see its docs.
//!
//! Generated code names `raads` by the path the calling crate gives it, so a renamed
//! dependency (`aad = { package = "raads", .. }`) works.

#![warn(missing_docs)]

use proc_macro::TokenStream;
use proc_macro_crate::FoundCrate;
use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote};
use raads_codegen::emit::{self, Kind};
use raads_codegen::{Group, Signature};
use syn::spanned::Spanned;
use syn::{Error, FnArg, ItemFn, Pat, Result, ReturnType, Type};

mod front;
mod model;
use front::{Builder, Param, Shape};

/// Compile every function of an inherent `impl` block as a kernel (like `#[kernel]`) and
/// implement the trait of the engine module `path` from them: `#[kernels(path)]`.
///
/// The functions stay callable as written; the engine's `path::__raads_model!` checks them
/// against its specification (names, shapes, `#[data]`) and assembles the trait impl.
/// Consts the engine declares are forwarded from the block, so kernels can read them as
/// `Self::NAME` without the trait in scope. The generic parameter name `R` is reserved.
///
/// An engine provides `__raads_model!` as a `macro_rules!` that forwards its input to
/// `raads::__private::check_kernels!` with its spec (`fn` signatures over `N` with named
/// array lengths, and consts) and a callback that assembles the impl; see the `paths` and
/// `pde1d` modules of `raads-engines`.
#[proc_macro_attribute]
pub fn kernels(attr: TokenStream, item: TokenStream) -> TokenStream {
    let engine = match syn::parse::<syn::Path>(attr) {
        Ok(p) => p,
        Err(e) => {
            return Error::new(e.span(), "expected the engine module, e.g. `#[kernels(raads_engines::paths)]`").to_compile_error().into();
        }
    };
    let imp = syn::parse_macro_input!(item as syn::ItemImpl);
    model::kernels(engine, imp).unwrap_or_else(|e| e.to_compile_error()).into()
}

/// Check the kernels `#[kernels]` passed to an engine against the engine's specification
/// and call the engine back with them (see the `model` module). Not public API.
#[doc(hidden)]
#[proc_macro]
pub fn check_kernels(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as model::Input);
    model::check(input).unwrap_or_else(|e| e.to_compile_error()).into()
}

/// Compile a kernel.
///
/// ```ignore
/// #[kernel]
/// fn f<N: AdNum>(v: [N; 2], #[data] z: N) -> N {
///     (v[0] * v[1]).exp() + (v[0] * v[0] + z).sqrt() / v[1]
/// }
/// ```
///
/// Keeps the function (callable on any `AdNum`: `f64`, SIMD packs, `Dual`, the tracer) and
/// adds a module of the same name with `eval`, `vjp`, `fwd`, `bwd`, `NRES` and `graph`.
/// Parameters are `N` or `[N; LITERAL]`; `#[data]` marks non-differentiable ones.
/// The body is `let` bindings, assignments and a final value using `+ - * /`, `exp ln sqrt
/// powf relu detach maximum minimum smooth_relu`, `N::select_gt(a, b, x, y)`, `N::from(k)`,
/// integer literal indices and `f64` constants: literals, paths (`Self::EPS`,
/// `std::f64::consts::PI`) and SCREAMING_CASE names; any other bare name must be a parameter
/// or a local. Integer literals are `f64`s (`x * 2`), and plain `f64`s are accepted where
/// an `N` is expected (`x.maximum(0.0)`). Parameter names `y`, `res`, `a_*`, and `v`, `a` or
/// `q` followed by digits are reserved by the generated code, as is the generic parameter name
/// `R` on a `#[kernels]` `impl`.
///
/// The generated code refers to `raads` by the name the calling crate's `Cargo.toml` gives
/// it, so a renamed dependency (`aad = { package = "raads", .. }`) works.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return Error::new(Tokens::from(attr).span(), "`#[kernel]` takes no arguments").to_compile_error().into();
    }
    let f = syn::parse_macro_input!(item as ItemFn);
    kernel_fn(f).unwrap_or_else(|e| e.to_compile_error()).into()
}

/// A compiled kernel: the rewritten source function and the generated items.
pub(crate) struct Compiled {
    pub source: ItemFn,
    pub eval: Tokens,
    pub vjp: Tokens,
    pub fwd: Tokens,
    pub bwd: Tokens,
    pub nres: usize,
    pub graph: Tokens,
}

/// Path of the `raads` crate as the calling crate names it. Inside `raads` itself
/// (`FoundCrate::Itself`, also reported for its doctests) `::raads` still resolves: the crate
/// declares `extern crate self as raads`.
pub(crate) fn krate() -> Tokens {
    match proc_macro_crate::crate_name("raads") {
        Ok(FoundCrate::Name(name)) => {
            let id = format_ident!("{name}");
            quote!(::#id)
        }
        _ => quote!(::raads),
    }
}

fn kernel_fn(f: ItemFn) -> Result<Tokens> {
    let krate = krate();
    let c = compile(f, "pub", ["eval", "vjp", "fwd", "bwd"], &krate)?;
    let Compiled { source, eval, vjp, fwd, bwd, nres, graph, .. } = c;
    let vis = &source.vis;
    let name = &source.sig.ident;
    let doc = format!("Compiled forward and reverse code of [`{name}`](fn@{name}).");
    Ok(quote! {
        #source

        #[doc = #doc]
        #[allow(clippy::all, unused_imports)]
        #vis mod #name {
            use super::*;
            /// Residuals `fwd` saves per call.
            pub const NRES: usize = #nres;
            #eval
            #vjp
            #fwd
            #bwd
            /// The kernel graph (for tests against the tracer).
            #[doc(hidden)]
            pub fn graph() -> #krate::__private::codegen::Graph { #graph }
        }
    })
}

fn shape(ty: &Type, n: &syn::Ident) -> Result<Shape> {
    let is_n = |t: &Type| matches!(t, Type::Path(p) if p.qself.is_none() && p.path.is_ident(n));
    match ty {
        t if is_n(t) => Ok(Shape::Scalar),
        Type::Array(a) if is_n(&a.elem) => {
            let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(k), .. }) = &a.len else {
                return Err(Error::new(a.len.span(), "kernel array lengths must be integer literals"));
            };
            Ok(Shape::Array(k.base10_parse()?))
        }
        Type::Paren(p) => shape(&p.elem, n),
        _ => Err(Error::new(ty.span(), format!("kernel parameters and results are `{n}` or `[{n}; LEN]`"))),
    }
}

/// Parse, build the graph and emit. `names` are the generated eval/vjp/fwd/bwd names;
/// `krate` is the path of `raads`.
pub(crate) fn compile(mut f: ItemFn, vis: &str, names: [&str; 4], krate: &Tokens) -> Result<Compiled> {
    let tys: Vec<_> = f.sig.generics.type_params().collect();
    if tys.len() != 1 {
        return Err(Error::new(f.sig.generics.span(), "a kernel has exactly one type parameter, e.g. `<N: AdNum>`"));
    }
    let n = tys[0].ident.clone();
    let mut params = Vec::new();
    for arg in f.sig.inputs.iter_mut() {
        let FnArg::Typed(pt) = arg else {
            return Err(Error::new(arg.span(), "kernels are free functions or associated functions without `self`"));
        };
        let data = pt.attrs.iter().any(|a| a.path().is_ident("data"));
        pt.attrs.retain(|a| !a.path().is_ident("data"));
        let Pat::Ident(id) = &*pt.pat else { return Err(Error::new(pt.pat.span(), "parameters must be plain names")) };
        let name = id.ident.to_string();
        if is_reserved(&name) {
            return Err(Error::new(
                id.span(),
                format!(
                    "parameter name `{name}` is reserved by the generated code (reserved: `y`, `res`, `a_*`, `v`, `a` or `q` followed by digits)"
                ),
            ));
        }
        params.push(Param { name, shape: shape(&pt.ty, &n)?, data });
    }
    let ret = match &f.sig.output {
        ReturnType::Type(_, t) => shape(t, &n)?,
        ReturnType::Default => return Err(Error::new(f.sig.span(), "a kernel returns `N` or `[N; LEN]`")),
    };
    let mut b = Builder::new(&params, &n, krate);
    let (val, body) = b.block(&f.block)?;
    b.outputs(&val, ret, f.block.span())?;
    let sig = Signature {
        groups: params
            .iter()
            .map(|p| Group {
                name: p.name.clone(),
                len: match p.shape {
                    Shape::Scalar => None,
                    Shape::Array(k) => Some(k),
                },
                data: p.data,
            })
            .collect(),
        out: match ret {
            Shape::Scalar => None,
            Shape::Array(k) => Some(k),
        },
    };
    let emit_fn = |kind, name: &str| -> Result<Tokens> {
        emit::function(&b.g, &sig, kind, vis, name, &krate.to_string())
            .parse()
            .map_err(|e| Error::new(f.sig.span(), format!("internal error: generated code does not parse: {e}")))
    };
    let eval = emit_fn(Kind::Eval, names[0])?;
    let vjp = emit_fn(Kind::Vjp, names[1])?;
    let fwd = emit_fn(Kind::Fwd, names[2])?;
    let bwd = emit_fn(Kind::Bwd, names[3])?;
    let nres = emit::nres(&b.g);
    let graph = b.graph_code();
    *f.block = syn::parse2(quote!({ #body }))?;
    Ok(Compiled { source: f, eval, vjp, fwd, bwd, nres, graph })
}

/// Names the generated code uses next to the parameters: `res` (residuals), `a_y` and
/// `a_<param>` (adjoints), `v<i>` and `a<i>` (node values and adjoints), `q<i>` (quotients).
fn is_reserved(s: &str) -> bool {
    let temp = s.len() > 1 && s.starts_with(['v', 'a', 'q']) && s[1..].bytes().all(|c| c.is_ascii_digit());
    s == "y" || s == "res" || s.starts_with("a_") || temp
}
