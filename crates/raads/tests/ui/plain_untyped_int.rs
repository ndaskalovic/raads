use raads::{AdNum, kernel};

const IDX: [usize; 2] = [1, 0];

#[kernel]
fn k<N: AdNum>(x: [N; 2]) -> N {
    let mut s = x[0];
    for i in 0..2 {
        let k = IDX[i];
        s = s * x[k];
    }
    s
}

fn main() {}
