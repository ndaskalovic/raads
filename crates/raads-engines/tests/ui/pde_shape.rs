use raads::{AdNum, kernels};

pub struct Model;

#[kernels(raads_engines::pde1d)]
impl Model {
    const OBSTACLE: bool = true;

    pub fn coeffs<N: AdNum>(#[data] _x: N, p: [N; 2]) -> [N; 2] {
        [p[0], p[1]]
    }

    pub fn initial<N: AdNum>(x: N, _p: [N; 2]) -> N {
        x.exp()
    }

    pub fn x0<N: AdNum>(p: [N; 2]) -> N {
        p[0]
    }

    pub fn horizon<N: AdNum>(p: [N; 2]) -> N {
        p[1]
    }
}

fn main() {}
