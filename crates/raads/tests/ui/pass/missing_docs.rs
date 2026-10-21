//! Generated items are documented.
#![deny(missing_docs)]

use raads::{AdNum, kernel};

/// A kernel.
#[kernel]
pub fn hinge<N: AdNum>(s: [N; 1], #[data] k: N) -> N {
    (s[0].exp() - k).relu()
}

fn main() {}
