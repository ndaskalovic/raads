use raads::{AdNum, kernel};

#[kernel]
fn k<N: AdNum>(x: [N; 2]) -> N {
    let a = [x[0]; 2];
    let mut b = a;
    for i in 0..2 {
        b[i] = b[i] * x[1];
    }
    a[0] + b[0]
}

fn main() {}
