//! A graph with loops as the straight-line graph that executing it records.

use crate::{Graph, NONE, Op};

/// Runs the loops, array accesses and carried values of `g` at expansion time and records
/// the remaining operations in execution order: the graph the runtime tracer records for
/// the same kernel. Panics on an index out of bounds.
pub fn unroll(g: &Graph) -> Graph {
    let mut u = Unroll {
        g,
        ends: g.ends(),
        parents: g.parents(),
        out: Graph::default(),
        map: vec![NONE; g.len()],
        arrays: vec![Vec::new(); g.len()],
        cur: vec![NONE; g.len()],
    };
    u.run(0, g.len(), &mut Vec::new());
    u.out.outs = g.outs.iter().map(|&o| u.map[o]).collect();
    u.out
}

struct Unroll<'a> {
    g: &'a Graph,
    ends: Vec<usize>,
    parents: Vec<usize>,
    out: Graph,
    /// Node of `out` holding the current value of each node of `g`.
    map: Vec<usize>,
    /// Elements of each local array.
    arrays: Vec<Vec<usize>>,
    /// Current value of each carried value.
    cur: Vec<usize>,
}

impl Unroll<'_> {
    fn run(&mut self, from: usize, to: usize, vars: &mut Vec<(String, f64)>) {
        let mut i = from;
        while i < to {
            let n = &self.g.nodes[i];
            let arg = |x: usize| if x == NONE { NONE } else { self.map[x] };
            match &n.op {
                Op::Loop(var, lo, hi) => {
                    let end = self.ends[i];
                    let (lo, hi) = (lo.eval(vars).expect("loop bound"), hi.eval(vars).expect("loop bound"));
                    for p in (i + 1..end).filter(|&p| self.parents[p] == i && self.g.nodes[p].op == Op::Phi) {
                        self.cur[p] = self.map[self.g.nodes[p].a];
                    }
                    // locals of an iteration go out of scope at its end
                    let scope = vars.len();
                    for k in lo..hi {
                        vars.push((var.clone(), k as f64));
                        self.run(i + 1, end, vars);
                        vars.truncate(scope);
                    }
                    i = end + 1;
                    continue;
                }
                Op::End => unreachable!("loops are run whole"),
                Op::Phi => self.map[i] = self.cur[i],
                Op::Next => self.cur[n.a] = self.map[n.b],
                Op::Out => self.map[i] = self.cur[n.a],
                Op::Array(len) => self.arrays[i] = vec![self.map[n.a]; *len],
                Op::Load(idx) => self.map[i] = self.arrays[n.a][idx.eval(vars).expect("index")],
                Op::Store(idx) => {
                    let k = idx.eval(vars).expect("index");
                    self.arrays[n.a][k] = self.map[n.b];
                }
                Op::LetInt(idx) => vars.push((format!("p{i}"), idx.eval(vars).expect("integer") as f64)),
                Op::LetNum(x) => vars.push((format!("p{i}"), x.call(vars).expect("number"))),
                Op::Num(x) => {
                    let k = x.call(vars).expect("number");
                    self.map[i] = self.out.push(Op::Const(crate::f64_literal(k)), NONE, NONE, NONE, NONE);
                }
                Op::Param { first, len, data, idx } => {
                    let k = idx.eval(vars).expect("index");
                    assert!(k < *len, "index {k} out of bounds ({len})");
                    let leaf = if *data { Op::Data(first + k) } else { Op::Input(first + k) };
                    let j = self.g.nodes.iter().position(|m| m.op == leaf).expect("parameter leaf");
                    self.map[i] = self.map[j];
                }
                op => self.map[i] = self.out.push(op.clone(), arg(n.a), arg(n.b), arg(n.c), arg(n.d)),
            }
            i += 1;
        }
    }
}
