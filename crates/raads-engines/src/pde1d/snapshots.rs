//! Multi-horizon solve: one solve saves the solution at several horizons.
//!
//! Same PDE, operator and scheme as [`gradient`](super::gradient) (theta scheme, LU
//! factorised per time block, cell-averaged initial condition, obstacle after each solve,
//! `u_xx = 0` boundaries). Differences:
//! - The time grid is a list of blocks so every horizon lands exactly on a step. The first
//!   block starts with Rannacher steps (implicit half-steps) that damp kinks in the initial
//!   condition, which matters for short horizons.
//! - The solution u and its time derivative `u_τ = L u` (0 where the obstacle is active) are
//!   saved at every horizon instead of being interpolated once at x0. The model's `x0` and
//!   `horizon` are unused; the time grid is data, not differentiated.
//!
//! With [`Dual`](raads::Dual) parameters one pass gives values and one forward tangent per
//! lane at every node. [`snapshots_weighted`] adds a reverse sweep for a weighted sum of
//! reads: its gradient w.r.t. every model parameter in one sweep.

use super::{PdeModel, Scheme};
use raads::{AdNum, Arr, Lanes, Real};

/// Spatial grid and time stepping of a multi-horizon solve.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SnapshotGrid {
    /// Lowest node.
    pub x_lo: f64,
    /// Highest node.
    pub x_hi: f64,
    /// Space nodes.
    pub nx: usize,
    /// Largest time step; each horizon segment gets equal steps no longer than this.
    pub dt: f64,
    /// Crank–Nicolson steps replaced by two implicit half-steps each at the start.
    pub rannacher: usize,
}

impl SnapshotGrid {
    /// With two Rannacher steps.
    pub fn new(x_lo: f64, x_hi: f64, nx: usize, dt: f64) -> Self {
        SnapshotGrid { x_lo, x_hi, nx, dt, rannacher: 2 }
    }
}

#[derive(Clone, Copy, Debug)]
struct Block {
    dt: f64,
    th: f64,
    steps: usize,
    /// Global time level at the start of the block.
    level0: usize,
    /// Horizon index reached at the end of the block.
    horizon: Option<usize>,
}

/// Each horizon segment gets `ceil(len / dt)` equal steps (at least `rannacher + 1`); the
/// first `rannacher` Crank–Nicolson steps become `2 * rannacher` implicit half-steps.
fn schedule(horizons: &[f64], dt_target: f64, rannacher: usize) -> Vec<Block> {
    let mut bl = Vec::new();
    let (mut t_prev, mut lev) = (0.0, 0);
    for (m, &t) in horizons.iter().enumerate() {
        let seg = t - t_prev;
        assert!(seg > 0.0, "horizons must be positive and strictly increasing");
        let ns = (rannacher + 1).max((seg / dt_target).ceil() as usize);
        let dt = seg / ns as f64;
        let mut start = 0;
        if m == 0 && rannacher > 0 {
            bl.push(Block { dt: dt * 0.5, th: 1.0, steps: 2 * rannacher, level0: lev, horizon: None });
            lev += 2 * rannacher;
            start = rannacher;
        }
        bl.push(Block { dt, th: 0.5, steps: ns - start, level0: lev, horizon: Some(m) });
        lev += ns - start;
        t_prev = t;
    }
    bl
}

/// A weighted read `weight · u(x, T_horizon)`, per lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Read<L> {
    /// Horizon index.
    pub horizon: usize,
    /// Grid coordinate.
    pub x: L,
    /// Weight.
    pub weight: L,
}

/// `u(x, T_m)` and `u_τ(x, T_m)` on the grid for every horizon `m`.
#[derive(Clone, Debug)]
#[must_use]
pub struct Snapshots<R> {
    x_lo: f64,
    h: f64,
    nx: usize,
    slices: Vec<R>,
    /// `u_τ = a u_xx + b u_x - c u` (discrete `L u`); 0 on nodes held by the obstacle.
    taus: Vec<R>,
}

impl<R: Real> Snapshots<R> {
    /// Lowest grid coordinate.
    pub fn x_lo(&self) -> f64 {
        self.x_lo
    }

    /// Node spacing.
    pub fn h(&self) -> f64 {
        self.h
    }

    /// Space nodes.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Nearest interior node per lane (at least two nodes from either end) and the offset
    /// `t = (x - x_j) / h`.
    fn weights(&self, x: R::Lanes) -> (Vec<usize>, R::Lanes) {
        let j: Vec<usize> =
            (0..R::Lanes::W).map(|l| (((x.lane(l) - self.x_lo) / self.h + 0.5) as i64).clamp(2, self.nx as i64 - 3) as usize).collect();
        let xj = R::Lanes::from_fn(|l| self.x_lo + j[l] as f64 * self.h);
        (j, (x - xj) / self.h)
    }

    /// u (quadratic through the nearest 3 nodes), u_x (its slope) and u_xx (second differences
    /// at the nearest node and its neighbour on x's side, linearly interpolated: O(h²)).
    fn read_buf(&self, buf: &[R], m: usize, x: R::Lanes) -> [R; 3] {
        let (j, ts) = self.weights(x);
        let t = R::lift(ts);
        let at = R::lift(R::Lanes::select_gt(ts, R::Lanes::from(0.0), ts, -ts));
        let s = &buf[m * self.nx..(m + 1) * self.nx];
        let side = |l: usize| if ts.lane(l) >= 0.0 { 1 } else { -1 };
        let at_off = |off: isize| R::per_lane(|l| s[(j[l] as isize + off) as usize]);
        let (vm, vc, vp) = (at_off(-1), at_off(0), at_off(1));
        let vi = R::per_lane(|l| s[(j[l] as isize + side(l)) as usize]); // inner neighbour on x's side
        let vo = R::per_lane(|l| s[(j[l] as isize + 2 * side(l)) as usize]); // outer node on x's side
        let h2 = self.h * self.h;
        let d2c = (vm - vc * 2.0 + vp) / h2;
        let d2n = (vc - vi * 2.0 + vo) / h2;
        [
            t * (t - 1.0) * 0.5 * vm + (t * t).rsub(1.0) * vc + t * (t + 1.0) * 0.5 * vp,
            ((t - 0.5) * vm - t * 2.0 * vc + (t + 0.5) * vp) / self.h,
            at.rsub(1.0) * d2c + at * d2n,
        ]
    }

    /// u, u_x, u_xx at horizon index `m`, per-lane coordinate `x`. Reads use the nodes nearest
    /// `x` at least two from either end, so an `x` near or outside the grid edge is extrapolated.
    /// Panics if `m` is out of range.
    pub fn read(&self, m: usize, x: R::Lanes) -> [R; 3] {
        self.read_buf(&self.slices, m, x)
    }

    /// The same read of `u_τ` (`u_τ = L u` on nodes where the obstacle is inactive, else 0):
    /// u_τ, u_τx, u_τxx. No dt error.
    pub fn read_tau(&self, m: usize, x: R::Lanes) -> [R; 3] {
        self.read_buf(&self.taus, m, x)
    }
}

/// Forward solve saving every horizon; with `GRADS`, also the reverse sweep of
/// `Σ_k weight_k u(x_k, T_{horizon_k})` into `a_p`.
fn solve<M: PdeModel, R: Real, const GRADS: bool>(
    p: &M::P<R>,
    g: &SnapshotGrid,
    horizons: &[f64],
    reads: &[Read<R::Lanes>],
    a_p: &mut M::P<R>,
) -> Snapshots<R> {
    let nx = g.nx;
    assert!(nx >= 5, "need at least 5 grid nodes");
    assert!(g.dt.is_finite() && g.dt > 0.0, "need a finite time step dt > 0, got {}", g.dt);
    assert!(!horizons.is_empty(), "need at least 1 horizon");
    assert!(reads.iter().all(|pt| pt.horizon < horizons.len()), "Read::horizon out of range");
    let mut sc = Scheme::<M, R>::new(p, g.x_lo, g.x_hi, nx, GRADS);
    let bl = schedule(horizons, g.dt, g.rannacher);
    let last = bl[bl.len() - 1];
    let mut ys = vec![R::from(0.0); if GRADS { (last.level0 + last.steps + 1) * nx } else { 0 }];
    let zero = R::from(0.0);
    let mut snaps = Snapshots { x_lo: g.x_lo, h: sc.h, nx, slices: vec![zero; horizons.len() * nx], taus: vec![zero; horizons.len() * nx] };
    if GRADS {
        ys[..nx].copy_from_slice(&sc.cur);
    }

    // forward: per block factorise once, step, save u and u_τ at each horizon
    for b in &bl {
        sc.factor(R::from(b.dt), b.th);
        for s in 0..b.steps {
            if GRADS {
                let lev = b.level0 + s + 1;
                sc.step::<true>(&mut ys[lev * nx..(lev + 1) * nx]);
            } else {
                sc.step::<false>(&mut []);
            }
        }
        if let Some(m) = b.horizon {
            let (cur, l, dd, u, obst) = (&sc.cur, &sc.l, &sc.dd, &sc.u, &sc.obst);
            snaps.slices[m * nx..(m + 1) * nx].copy_from_slice(cur);
            for i in 0..nx {
                let c = cur[i];
                let mut lu = dd[i] * c;
                if i > 0 {
                    lu = l[i].mul_add(cur[i - 1], lu);
                }
                if i < nx - 1 {
                    lu = u[i].mul_add(cur[i + 1], lu);
                }
                snaps.taus[m * nx + i] = if M::OBSTACLE { R::select_gt(c, obst[i], lu, zero) } else { lu };
            }
        }
    }

    if GRADS {
        for bi in (0..bl.len()).rev() {
            let b = bl[bi];
            if bi + 1 < bl.len() {
                sc.rescale(R::from(bl[bi + 1].dt / b.dt)); // see `Scheme::rescale`
            }
            // seed the reads of this horizon (aV is the adjoint of V at the block's last level)
            if let Some(m) = b.horizon {
                for pt in reads.iter().filter(|pt| pt.horizon == m) {
                    let (j, t) = snaps.weights(pt.x);
                    let w = [t * (t - 1.0) * 0.5 * pt.weight, (t * t).rsub(1.0) * pt.weight, t * (t + 1.0) * 0.5 * pt.weight];
                    for (lane, &jl) in j.iter().enumerate() {
                        for (k, wk) in w.iter().enumerate() {
                            let node = jl + k - 1;
                            let q = sc.a_v[node];
                            let upd = q + R::lift(R::Lanes::from(wk.lane(lane)));
                            sc.a_v[node] = R::per_lane(|l| if l == lane { upd } else { q });
                        }
                    }
                }
            }
            sc.factor(R::from(b.dt), b.th);
            for s in (0..b.steps).rev() {
                sc.step_back(&ys, b.level0 + s, b.th);
            }
        }
        sc.rescale(R::from(bl[0].dt));
        sc.to_params(p, a_p);
    }
    snaps
}

/// The solution at every horizon in one solve.
///
/// # Panics
///
/// If `grid.nx < 5`, `grid.dt` is not finite and positive, `x_lo`/`x_hi` are not finite with
/// `x_lo < x_hi`, or `horizons` is empty or not positive and strictly increasing.
pub fn snapshots<M: PdeModel, R: Real>(params: &M::P<R>, grid: &SnapshotGrid, horizons: &[f64]) -> Snapshots<R> {
    let mut a: M::P<R> = Arr::splat(R::from(0.0));
    solve::<M, R, false>(params, grid, horizons, &[], &mut a)
}

/// A weighted sum of reads and its gradient, from [`snapshots_weighted`].
#[derive(Clone, Debug)]
#[non_exhaustive]
#[must_use]
pub struct WeightedSnapshots<R, G> {
    /// `Σ_k weight_k u(x_k, T_{horizon_k})`, per lane.
    pub value: R,
    /// d value / d params, per lane.
    pub grad: G,
    /// The solution the reads use.
    pub snapshots: Snapshots<R>,
}

/// A weighted sum of reads `Σ_k weight_k u(x_k, T_{horizon_k})` and its gradient w.r.t. every
/// model parameter in one reverse sweep (per lane). With [`Dual`](raads::Dual) the gradient
/// tangents are Hessian-vector products of the sum.
///
/// # Panics
///
/// As [`snapshots`], or if a read's horizon index is out of range.
pub fn snapshots_weighted<M: PdeModel, R: Real>(
    params: &M::P<R>,
    grid: &SnapshotGrid,
    horizons: &[f64],
    reads: &[Read<R::Lanes>],
) -> WeightedSnapshots<R, M::P<R>> {
    let mut grad: M::P<R> = Arr::splat(R::from(0.0));
    let snapshots = solve::<M, R, true>(params, grid, horizons, reads, &mut grad);
    let value = reads.iter().fold(R::from(0.0), |v, pt| v + R::lift(pt.weight) * snapshots.read(pt.horizon, pt.x)[0]);
    WeightedSnapshots { value, grad, snapshots }
}
