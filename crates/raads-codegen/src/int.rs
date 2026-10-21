//! Integer expressions: loop bounds and array indices.

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
}

impl Int {
    /// Value with the loop variables `vars` (name, value); `None` on overflow, a division
    /// by zero or an unknown variable.
    pub fn eval(&self, vars: &[(String, usize)]) -> Option<usize> {
        match self {
            Int::Lit(k) => Some(*k),
            Int::Var(v) => vars.iter().rev().find(|(n, _)| n == v).map(|&(_, k)| k),
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
        }
    }
    /// Literal value, if the expression has no variables.
    pub fn constant(&self) -> Option<usize> {
        self.eval(&[])
    }
}

/// Rust source (`usize`), fully parenthesised.
impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Int::Lit(k) => write!(f, "{k}usize"),
            Int::Var(v) => write!(f, "{v}"),
            Int::Bin(a, op, b) => write!(f, "({a} {op} {b})"),
        }
    }
}
