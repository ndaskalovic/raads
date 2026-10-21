use raads::{AdNum, kernel};

const K: f64 = 1.0;

#[kernel]
fn k<N: AdNum>(x: N) -> N {
    K.maximum(x)
}

fn main() {}
