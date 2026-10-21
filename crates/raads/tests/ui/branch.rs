use raads::{AdNum, kernel};

#[kernel]
fn k<N: AdNum>(x: N, #[data] z: N) -> N {
    if z > N::from(0.0) { x } else { -x }
}

fn main() {}
