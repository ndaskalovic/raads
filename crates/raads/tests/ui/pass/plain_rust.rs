//! Plain Rust indices and numbers compile without warnings.
#![deny(warnings)]

use raads::{AdNum, kernel};

const MATURITIES: [usize; 3] = [2, 0, 1];
const RATES: [f64; 3] = [0.01, 0.02, 0.03];

fn weight(h: usize) -> f64 {
    h as f64 * 0.5
}

/// A kernel.
#[kernel]
pub fn book<N: AdNum>(x: [N; 3], #[data] z: [N; 3]) -> N {
    let mut v = N::from(0.0);
    for i in 0..3 {
        let m: usize = MATURITIES[i];
        v = v + x[m] * z[(m + i) % 3] * (RATES[i] + weight(m + 1));
    }
    v
}

fn main() {}
