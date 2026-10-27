use raads::{AdNum, kernel};

#[kernel]
fn k<N: AdNum>(x: [N; 3]) -> N {
    let mut s = x[0];
    for i in 0..2 {
        let mut j: usize = i;
        j = j + 1;
        s = s * x[j];
    }
    s
}

fn main() {}
