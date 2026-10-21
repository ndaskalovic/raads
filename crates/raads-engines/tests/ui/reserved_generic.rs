use raads::{AdNum, kernels};
use std::marker::PhantomData;

pub struct Model<R>(PhantomData<R>);

#[kernels(raads_engines::paths)]
impl<R> Model<R> {
    pub fn setup<N: AdNum>(p: [N; 1], #[data] n_steps: N) -> [N; 1] {
        [p[0] / n_steps]
    }

    pub fn init<N: AdNum>(p: [N; 1]) -> [N; 1] {
        p
    }

    pub fn step<N: AdNum>(s: [N; 1], c: [N; 1], #[data] z: [N; 1]) -> [N; 1] {
        [s[0] + c[0] * z[0]]
    }

    pub fn functional<N: AdNum>(s: [N; 1], _p: [N; 1], _c: [N; 1]) -> N {
        s[0]
    }
}

fn main() {}
