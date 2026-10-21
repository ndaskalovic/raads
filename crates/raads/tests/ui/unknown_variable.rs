use raads::{AdNum, kernel};

#[kernel]
fn k<N: AdNum>(x: [N; 2]) -> N {
    let xx = x[0] * x[1];
    xx + yy
}

fn main() {}
