use raads::{AdNum, kernel};

#[kernel]
fn k<N: AdNum>(x: [N; 1]) -> N {
    let mut c = 1.0;
    for _i in 0..3 {
        c = c * 2.0;
    }
    x[0] * c
}

fn main() {}
