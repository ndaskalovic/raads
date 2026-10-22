use raads::{AdNum, kernel};

const LEN: usize = 3;

#[kernel]
fn k<N: AdNum>(x: [N; 3]) -> N {
    let mut s = x[0];
    for i in 0..LEN {
        s = s * x[i];
    }
    s
}

fn main() {}
