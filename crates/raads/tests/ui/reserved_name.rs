use raads::{AdNum, kernel};

#[kernel]
fn k<N: AdNum>(x: N, y: N) -> N {
    x * y
}

fn main() {}
