//! Helpers shared by the integration tests (each test crate uses a subset).
#![allow(dead_code)]

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// `n` standard normal draws.
pub fn normals(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| StandardNormal.sample(&mut rng)).collect()
}

/// `|a - b| / max(|b|, 1e-9)`.
pub fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-9)
}

/// Failed checks, collected so one run reports all of them.
#[derive(Default)]
pub struct Checks(pub Vec<String>);

impl Checks {
    /// `v <= tol`; NaN fails.
    pub fn le(&mut self, label: &str, v: f64, tol: f64) {
        println!("    {label:<48} {v:>12.4e} <= {tol:e}");
        if v.is_nan() || v > tol {
            self.0.push(format!("{label}: {v:e} > {tol:e}"));
        }
    }
    pub fn rel(&mut self, label: &str, v: f64, r: f64, tol: f64) {
        self.le(&format!("{label} ({v} vs {r}), rel"), rel(v, r), tol);
    }
    pub fn abs(&mut self, label: &str, v: f64, r: f64, tol: f64) {
        self.le(&format!("{label} ({v} vs {r}), abs"), (v - r).abs(), tol);
    }
    pub fn assert_ok(&self) {
        assert!(self.0.is_empty(), "{:#?}", self.0);
    }
}
