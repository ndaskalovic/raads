use raads::{AdNum, kernel};

#[kernel]
fn k<N: AdNum>(x: Vec<N>) -> N {
    x[0]
}

fn main() {}
