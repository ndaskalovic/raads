//! Graph analyses.

use crate::{Graph, Op};

/// Node depends on a differentiable input.
pub fn active(g: &Graph) -> Vec<bool> {
    let mut act = vec![false; g.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        act[i] = match n.op {
            Op::Input(_) => true,
            Op::Stop => false,
            _ if n.is_leaf() => false,
            _ => n.args().any(|a| act[a]),
        };
    }
    act
}

/// Node reaches an output through a differentiable operand (the mask operands of
/// `SelGt` and the operand of `Stop` receive no adjoint).
pub fn useful(g: &Graph) -> Vec<bool> {
    let mut use_ = vec![false; g.len()];
    for &o in &g.outs {
        use_[o] = true;
    }
    for i in (0..g.len()).rev() {
        let n = &g.nodes[i];
        if !use_[i] || n.is_leaf() || n.op == Op::Stop {
            continue;
        }
        if n.op == Op::SelGt {
            use_[n.c] = true;
            use_[n.d] = true;
        } else {
            for a in [n.a, n.b] {
                if a != crate::NONE {
                    use_[a] = true;
                }
            }
        }
    }
    use_
}

/// Node needs an adjoint: active and useful.
pub fn need(g: &Graph) -> Vec<bool> {
    active(g).iter().zip(useful(g)).map(|(&a, u)| a && u).collect()
}

/// Node value is needed by the forward evaluation of the outputs (all operands count).
pub fn live(g: &Graph) -> Vec<bool> {
    let mut l = vec![false; g.len()];
    for &o in &g.outs {
        l[o] = true;
    }
    for i in (0..g.len()).rev() {
        if l[i] {
            for a in g.nodes[i].args() {
                l[a] = true;
            }
        }
    }
    l
}

/// Forward values the reverse sweep reads.
pub fn reads(g: &Graph, need: &[bool]) -> Vec<bool> {
    let mut rd = vec![false; g.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        if !need[i] || n.is_leaf() {
            continue;
        }
        match n.op {
            Op::Mul => {
                if need[n.a] {
                    rd[n.b] = true;
                }
                if need[n.b] {
                    rd[n.a] = true;
                }
            }
            Op::Div => {
                if need[n.a] || need[n.b] {
                    rd[n.b] = true;
                }
                if need[n.b] {
                    rd[i] = true;
                }
            }
            Op::Exp | Op::Sqrt => rd[i] = true,
            Op::Ln | Op::Powk(_) | Op::Relu => rd[n.a] = true,
            Op::SelGt => {
                rd[n.a] = true;
                rd[n.b] = true;
            }
            _ => {}
        }
    }
    rd
}

/// Values `fwd` saves for `bwd`: read by the reverse sweep and not rematerialised
/// (data and constants are), in node order.
pub fn residuals(g: &Graph, reads: &[bool]) -> Vec<usize> {
    (0..g.len()).filter(|&i| reads[i] && !matches!(g.nodes[i].op, Op::Data(_) | Op::Const(_))).collect()
}

/// All analyses of one graph.
pub struct Analysis {
    /// See [`active`]. Inactive nodes depend only on data and constants: generated code
    /// computes them as passive `R::Lanes` packs.
    pub active: Vec<bool>,
    /// See [`need`].
    pub need: Vec<bool>,
    /// See [`live`].
    pub live: Vec<bool>,
    /// See [`reads`].
    pub reads: Vec<bool>,
    /// See [`residuals`].
    pub res: Vec<usize>,
}

impl Analysis {
    /// Run every pass.
    pub fn new(g: &Graph) -> Self {
        let need = need(g);
        let live = live(g);
        let reads = reads(g, &need);
        let res = residuals(g, &reads);
        Analysis { active: active(g), need, live, reads, res }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NONE;

    /// GBM log-space step: s + c0 + c1 * z. No residuals (z is rematerialised).
    #[test]
    fn gbm_step() {
        let mut g = Graph::default();
        let s = g.push(Op::Input(0), NONE, NONE, NONE, NONE);
        let c0 = g.push(Op::Input(1), NONE, NONE, NONE, NONE);
        let c1 = g.push(Op::Input(2), NONE, NONE, NONE, NONE);
        let z = g.push(Op::Data(0), NONE, NONE, NONE, NONE);
        let a = g.push(Op::Add, s, c0, NONE, NONE);
        let m = g.push(Op::Mul, c1, z, NONE, NONE);
        let y = g.push(Op::Add, a, m, NONE, NONE);
        g.outs.push(y);
        let an = Analysis::new(&g);
        assert_eq!(an.need, vec![true, true, true, false, true, true, true]);
        assert!(an.reads[z]);
        assert!(an.res.is_empty());
    }

    /// select_gt: mask operands get no adjoint but are read.
    #[test]
    fn selgt() {
        let mut g = Graph::default();
        let x = g.push(Op::Input(0), NONE, NONE, NONE, NONE);
        let y = g.push(Op::Input(1), NONE, NONE, NONE, NONE);
        let e = g.push(Op::Exp, x, NONE, NONE, NONE);
        let s = g.push(Op::SelGt, e, y, x, y);
        g.outs.push(s);
        let an = Analysis::new(&g);
        assert!(!an.need[e], "mask operand needs no adjoint");
        assert!(an.live[e], "mask operand is still evaluated");
        assert_eq!(an.res, vec![y, e]);
    }

    /// stop-gradient: the detached subgraph is evaluated but needs no adjoints or residuals.
    #[test]
    fn stop() {
        let mut g = Graph::default();
        let x = g.push(Op::Input(0), NONE, NONE, NONE, NONE);
        let e = g.push(Op::Exp, x, NONE, NONE, NONE);
        let s = g.push(Op::Stop, e, NONE, NONE, NONE);
        let y = g.push(Op::Mul, s, x, NONE, NONE);
        g.outs.push(y);
        let an = Analysis::new(&g);
        assert_eq!(an.need, vec![true, false, false, true]);
        assert!(an.live[e]);
        assert_eq!(an.res, vec![s], "only the detached value is saved, not exp(x)");
    }
}
