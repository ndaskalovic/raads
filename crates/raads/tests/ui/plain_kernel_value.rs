use raads::{AdNum, kernel};

fn weight(w: f64) -> f64 {
    w * 0.5
}

#[kernel]
fn k<N: AdNum>(x: [N; 2]) -> N {
    let y = x[0] * x[1];
    x[1] * weight(y)
}

fn main() {}
