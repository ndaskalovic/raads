use raads::{AdNum, kernels};

pub struct Model;

#[kernels(raads_engines::paths)]
impl Model {
    pub fn setup<N: AdNum>(p: [N; 3], #[data] n_steps: N) -> [N; 2] {
        let dt = p[2] / n_steps;
        [p[1] * dt, dt.sqrt()]
    }

    pub fn init<N: AdNum>(p: [N; 3]) -> [N; 1] {
        [p[0]]
    }

    pub fn step<N: AdNum>(s: [N; 1], c: [N; 2], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] + c[1] * z[0]]
    }

    pub fn functional<N: AdNum>(s: [N; 1], _p: [N; 3], _c: [N; 2]) -> N {
        s[0] * s[0]
    }

    pub fn helper<N: AdNum>(x: N) -> N {
        x * 2.0
    }
}

fn main() {}
