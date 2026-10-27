//! Graph analyses.

use crate::{Graph, Op};

/// Node depends on a differentiable input. A local array is active if any element stored in
/// it is, a carried value if its initial value or any update is.
pub fn active(g: &Graph) -> Vec<bool> {
    let mut act = vec![false; g.len()];
    let mut changed = true;
    while changed {
        changed = false;
        let mut set = |act: &mut Vec<bool>, j: usize| {
            if !act[j] {
                act[j] = true;
                changed = true;
            }
        };
        for (i, n) in g.nodes.iter().enumerate() {
            let a = match &n.op {
                Op::Input(_) => true,
                Op::Param { data, .. } => !data,
                Op::Stop | Op::Loop(..) | Op::End => false,
                _ if n.is_leaf() => false,
                Op::Next | Op::Store(_) => {
                    if act[n.b] {
                        set(&mut act, n.a);
                    }
                    false
                }
                Op::Phi | Op::Array(_) | Op::Load(_) | Op::Out => act[n.a],
                _ => n.args().any(|a| act[a]),
            };
            if a {
                set(&mut act, i);
            }
        }
    }
    act
}

/// Node reaches an output through a differentiable operand (the mask operands of
/// `SelGt` and the operand of `Stop` receive no adjoint). An element stored in a useful
/// array, and the update of a useful carried value, are useful.
pub fn useful(g: &Graph) -> Vec<bool> {
    let mut use_ = vec![false; g.len()];
    for &o in &g.outs {
        use_[o] = true;
    }
    let mut changed = true;
    while changed {
        changed = false;
        let mut set = |use_: &mut Vec<bool>, j: usize| {
            if j != crate::NONE && !use_[j] {
                use_[j] = true;
                changed = true;
            }
        };
        for i in (0..g.len()).rev() {
            let n = &g.nodes[i];
            match n.op {
                Op::Next | Op::Store(_) => {
                    if use_[n.a] {
                        set(&mut use_, n.b);
                    }
                }
                _ if !use_[i] || n.is_leaf() => {}
                Op::Stop | Op::Param { .. } | Op::Loop(..) | Op::End => {}
                Op::SelGt => {
                    set(&mut use_, n.c);
                    set(&mut use_, n.d);
                }
                _ => {
                    set(&mut use_, n.a);
                    set(&mut use_, n.b);
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
/// Loop markers and plain Rust locals (read by name, not as operands) are always live,
/// stores into live arrays and updates of live carried values too.
pub fn live(g: &Graph) -> Vec<bool> {
    let mut l = vec![false; g.len()];
    for &o in &g.outs {
        l[o] = true;
    }
    let mut changed = true;
    while changed {
        changed = false;
        for i in (0..g.len()).rev() {
            let n = &g.nodes[i];
            let keep = match n.op {
                Op::Loop(..) | Op::End | Op::LetInt(_) | Op::LetNum(_) | Op::LetConst(_) => true,
                Op::Next | Op::Store(_) => l[n.a],
                _ => l[i],
            };
            if !keep {
                continue;
            }
            for j in std::iter::once(i).chain(n.args()) {
                if !l[j] {
                    l[j] = true;
                    changed = true;
                }
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

/// Recomputed by the reverse sweep instead of saved: data, constants, plain Rust numbers,
/// and with `inputs` (a `vjp`, which has the inputs) elements of input arrays read with a
/// computed index.
pub fn remat(g: &Graph, i: usize, inputs: bool) -> bool {
    match g.nodes[i].op {
        Op::Data(_) | Op::Const(_) | Op::Num(_) => true,
        Op::Param { data, .. } => data || inputs,
        _ => false,
    }
}

/// Values `fwd` saves for `bwd` at the top level: read by the reverse sweep and not
/// rematerialised, in node order.
pub fn residuals(g: &Graph, reads: &[bool]) -> Vec<usize> {
    let parents = g.parents();
    (0..g.len()).filter(|&i| reads[i] && parents[i] == crate::NONE && !remat(g, i, false)).collect()
}

/// Total number of executions of each loop body ([`Op::Loop`] markers; 0 for other nodes).
/// Errors if a bound cannot be evaluated (overflow, division by zero). Walks the iterations of
/// every loop that contains loops; an innermost loop only adds its trip count.
pub fn runs(g: &Graph) -> Result<Vec<usize>, String> {
    fn go(g: &Graph, ends: &[usize], from: usize, to: usize, vars: &mut Vec<(String, f64)>, runs: &mut [usize]) -> Result<(), String> {
        let mut i = from;
        while i < to {
            if let Op::Loop(var, lo, hi) = &g.nodes[i].op {
                let bad = || format!("cannot evaluate the bounds `{lo}..{hi}` of loop `{var}`");
                let (lo, hi) = (lo.eval(vars).ok_or_else(bad)?, hi.eval(vars).ok_or_else(bad)?);
                if !g.nodes[i + 1..ends[i]].iter().any(|n| matches!(n.op, Op::Loop(..))) {
                    runs[i] += hi.saturating_sub(lo);
                    i = ends[i] + 1;
                    continue;
                }
                for k in lo..hi {
                    runs[i] += 1;
                    vars.push((var.clone(), k as f64));
                    go(g, ends, i + 1, ends[i], vars, runs)?;
                    vars.pop();
                }
                i = ends[i];
            }
            i += 1;
        }
        Ok(())
    }
    let mut runs = vec![0; g.len()];
    go(g, &g.ends(), 0, g.len(), &mut Vec::new(), &mut runs)?;
    Ok(runs)
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
    /// See [`Graph::parents`].
    pub parents: Vec<usize>,
    /// See [`Graph::ends`].
    pub ends: Vec<usize>,
    /// See [`runs`].
    pub runs: Vec<usize>,
}

impl Analysis {
    /// Run every pass.
    pub fn new(g: &Graph) -> Self {
        let need = need(g);
        let live = live(g);
        let reads = reads(g, &need);
        let res = residuals(g, &reads);
        let runs = runs(g).expect("loop bounds are checked by the front end");
        Analysis { active: active(g), need, live, reads, res, parents: g.parents(), ends: g.ends(), runs }
    }
    /// Values one iteration of `lp` saves on the tape (read by the reverse sweep, defined
    /// directly in the loop body, not rematerialised), in node order. `inputs`: see [`remat`].
    pub fn saved(&self, g: &Graph, lp: usize, inputs: bool) -> Vec<usize> {
        (lp + 1..self.ends[lp]).filter(|&i| self.reads[i] && self.parents[i] == lp && !remat(g, i, inputs)).collect()
    }
    /// Values of the loop body `lp` the reverse sweep recomputes at the start of an iteration,
    /// in node order: plain Rust locals, and parameter elements and plain Rust numbers it
    /// reads (constants are inlined where they are used).
    pub fn rematerialised(&self, g: &Graph, lp: usize, inputs: bool) -> Vec<usize> {
        (lp + 1..self.ends[lp]).filter(|&i| self.parents[i] == lp && self.recomputed(g, i, inputs)).collect()
    }
    /// The reverse sweep recomputes node `i` where its scope starts (see [`Self::rematerialised`]).
    pub fn recomputed(&self, g: &Graph, i: usize, inputs: bool) -> bool {
        match g.nodes[i].op {
            Op::LetInt(_) | Op::LetNum(_) | Op::LetConst(_) => true,
            Op::Param { .. } | Op::Num(_) => self.reads[i] && remat(g, i, inputs),
            _ => false,
        }
    }
    /// Length of the loop tape: saved values over all iterations of all loops.
    pub fn tape(&self, g: &Graph, inputs: bool) -> usize {
        (0..g.len()).filter(|&i| matches!(g.nodes[i].op, Op::Loop(..))).map(|l| self.runs[l] * self.saved(g, l, inputs).len()).sum()
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
