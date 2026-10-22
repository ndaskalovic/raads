//! Integer expressions (loop bounds and array indices) and plain Rust expressions.

use std::fmt;

/// `usize` arithmetic on literals and loop variables.
#[derive(Clone, Debug, PartialEq)]
pub enum Int {
    /// Literal.
    Lit(usize),
    /// Loop variable.
    Var(String),
    /// `a op b`, `op` one of `+ - * / %`.
    Bin(Box<Int>, char, Box<Int>),
    /// A plain Rust `usize` expression (`MATURITIES[i]`, `interval(h)`).
    Opaque(Opaque<usize>),
}

impl Int {
    /// Value with the loop variables `vars` (name, value); `None` on overflow, a division
    /// by zero, an unknown variable or a plain Rust expression without [`Opaque::eval`].
    pub fn eval(&self, vars: &[(String, usize)]) -> Option<usize> {
        match self {
            Int::Lit(k) => Some(*k),
            Int::Var(v) => lookup(vars, v),
            Int::Bin(a, op, b) => {
                let (a, b) = (a.eval(vars)?, b.eval(vars)?);
                match op {
                    '+' => a.checked_add(b),
                    '-' => a.checked_sub(b),
                    '*' => a.checked_mul(b),
                    '/' => a.checked_div(b),
                    '%' => a.checked_rem(b),
                    _ => None,
                }
            }
            Int::Opaque(o) => o.call(vars),
        }
    }
    /// Literal value, if the expression has no variables and no plain Rust parts.
    pub fn constant(&self) -> Option<usize> {
        self.eval(&[])
    }
    /// Has a plain Rust part, which the macro cannot evaluate.
    pub fn has_opaque(&self) -> bool {
        match self {
            Int::Lit(_) | Int::Var(_) => false,
            Int::Bin(a, _, b) => a.has_opaque() || b.has_opaque(),
            Int::Opaque(_) => true,
        }
    }
    /// Loop variables the expression reads.
    pub fn vars(&self) -> Vec<String> {
        match self {
            Int::Lit(_) => vec![],
            Int::Var(v) => vec![v.clone()],
            Int::Bin(a, _, b) => {
                let mut v = a.vars();
                v.extend(b.vars().into_iter().filter(|x| !v.contains(x)).collect::<Vec<_>>());
                v
            }
            Int::Opaque(o) => o.vars.clone(),
        }
    }
}

/// Rust source (`usize`), fully parenthesised.
impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Int::Lit(k) => write!(f, "{k}usize"),
            Int::Var(v) => write!(f, "{v}"),
            Int::Bin(a, op, b) => write!(f, "({a} {op} {b})"),
            Int::Opaque(o) => write!(f, "({})", o.src),
        }
    }
}

/// A plain Rust expression of the loop variables that the macro does not interpret
/// (`MATURITIES[i]`, `weight(h)`): the generated code evaluates its source where it is
/// used. A graph built at run time (a kernel's `graph()`) carries `eval`, which computes it
/// from the values of `vars`, so [`crate::unroll`] can run it.
#[derive(Clone, Debug)]
pub struct Opaque<T> {
    /// Rust source.
    pub src: String,
    /// Loop variables it reads.
    pub vars: Vec<String>,
    /// Its value given the values of `vars`, in order.
    pub eval: Option<fn(&[usize]) -> T>,
}

impl<T> Opaque<T> {
    /// Value with the loop variables `vars` (name, value).
    pub fn call(&self, vars: &[(String, usize)]) -> Option<T> {
        let vals: Option<Vec<usize>> = self.vars.iter().map(|v| lookup(vars, v)).collect();
        Some((self.eval?)(&vals?))
    }
}

/// Equal sources.
impl<T> PartialEq for Opaque<T> {
    fn eq(&self, o: &Self) -> bool {
        self.src == o.src && self.vars == o.vars
    }
}

/// Value of the innermost loop variable `v`.
fn lookup(vars: &[(String, usize)], v: &str) -> Option<usize> {
    vars.iter().rev().find(|(n, _)| n == v).map(|&(_, k)| k)
}
