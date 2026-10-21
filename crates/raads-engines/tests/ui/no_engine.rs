use raads::{AdNum, kernels};

pub struct Model;

#[kernels]
impl Model {
    pub fn x0<N: AdNum>(p: [N; 2]) -> N {
        p[0]
    }
}

fn main() {}
