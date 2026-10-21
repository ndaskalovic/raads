//! Fixed-size arrays as associated types of models (`type P<T: Copy> = [T; 5]`), so engines
//! stay generic over sizes on stable Rust.

/// `[T; N]` for some `N`. Sealed: implemented for arrays only.
pub trait Arr<T>: Copy + AsRef<[T]> + AsMut<[T]> + crate::sealed::Sealed {
    /// `N`.
    const LEN: usize;
    /// Every element `t`.
    fn splat(t: T) -> Self;
    /// Element `i` is `f(i)`.
    fn from_fn(f: impl FnMut(usize) -> T) -> Self;
}

impl<T, const N: usize> crate::sealed::Sealed for [T; N] {}

impl<T: Copy, const N: usize> Arr<T> for [T; N] {
    const LEN: usize = N;
    #[inline(always)]
    fn splat(t: T) -> Self {
        [t; N]
    }
    #[inline(always)]
    fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        std::array::from_fn(f)
    }
}
