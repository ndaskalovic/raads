use raads::{AdNum, kernel};

#[kernel]
fn k<N: AdNum>(x: [N; 1]) -> N {
    let mut s = x[0];
    for _i in 0..3 {
        s = s * x[0];
        let s = s + 1.0;
    }
    s
}

fn main() {}
