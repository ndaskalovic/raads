# Benchmarks

## XAD's benchmark suite

![Benchmark chart](ad-benchmarks/chart.png)

The Heston MC and SABR calibration workloads of [auto-differentiation/ad-benchmarks](https://github.com/auto-differentiation/ad-benchmarks), the suite in [XAD's README](https://github.com/auto-differentiation/xad#benchmarks), with raads added. Gradient time; Primal is the same workload in plain C++ doubles (the dashed line).

| # | Benchmark | Sensis | Primal | FD | XAD | Adept | autodiff | raads | raads SIMD |
|---|---|:-:|--:|--:|--:|--:|--:|--:|--:|
| 1 | Heston MC | 8 | 5.21 ms | 46.8 ms | 45.5 ms | 67.1 ms | DNF | 17.0 ms | **2.54 ms** |
| 2 | SABR Calib | 15 | 0.849 ms | 13.6 ms | 6.88 ms | 13.7 ms | 17.4 ms | 0.930 ms | **0.765 ms** |

- Same inputs, sizes and timing as the suite: its own code generates the random numbers and market data ([`ad-benchmarks/reference.cpp`](ad-benchmarks/reference.cpp)), median of 10 runs after 3 warm-ups, f64, single core. raads's values and gradients match XAD's to 1e-9.
- raads runs the suite's kernels ported line by line ([`src/bin/adbench.rs`](src/bin/adbench.rs)): Heston as a path model, the SABR objective as a kernel over strikes. raads: 1 lane; raads SIMD: 8 lanes (Heston), 4 (SABR).
- Apple M5, Apple clang 17. Differences from the suite's published run (Xeon, GCC 13, `-O3 -mavx2 -mfma`): `-mcpu=native` on ARM; CppAD left out (its build needs pkg-config); XAD-Codegen is commercial. The XVA and LIBOR workloads need loops in `#[kernel]`, not supported yet.
- Versions: ad-benchmarks c8439f9, XAD 9c33936, Adept d0a7751, autodiff b0a4fef.

Reproduce, from the repository root:

```bash
benchmarks/ad-benchmarks/run.sh   # CMAKE="uvx cmake" if cmake is not installed
```

It clones and builds the suite into `target/`, runs it, generates the inputs and XAD's results for raads, runs raads, and redraws `ad-benchmarks/chart.png` with the suite's plotting script ([`ad-benchmarks/plot.py`](ad-benchmarks/plot.py), raads added). Needs git, cmake, a C++17 compiler, uv and the pinned nightly.

## raads vs Enzyme and XAD

f64, single core, identical inputs (the same generator in Rust and C++, both on the system libm). raads, Enzyme and XAD agree to 10 significant digits.

- **Monte Carlo**: GBM European call, log-space exact scheme, 200k paths × 52 steps, price + 5 Greeks (S₀, r, σ, T, K). raads: `raads-engines::paths` with 16 f64 lanes. XAD: path-wise tape (record, adjoint to the setup mark, reset).
- **Function**: f(x, y; z) = exp(xy) + √(x² + z)/y, value and gradient at 1M points. raads: `#[kernel]`, 16 points per call. XAD: one recording per point.
- **Second order**: Hessian-vector products (H·v, forward-over-reverse), returned with the value and gradient, at 1M points. The function above, and SABR implied volatility (Hagan 2002, β = 0.5) w.r.t. (F, α, ρ, ν) at random parameters and out-of-the-money strikes. raads: the kernel on `Dual` lanes (8 wide). Enzyme: `#[autodiff_forward]` over the `#[autodiff_reverse]` function.

Enzyme runs in two versions: plain Rust with one attribute (the Monte Carlo pricer over blocks of 16 paths, the scalar function per point), and hand-vectorised with `std::simd` (16 lanes, explicit FMA, a SIMD `exp`).

Apple M5, 2026-10-20, best of 31 (two runs agree within 2.1%):

| | raads | Enzyme | Enzyme, `std::simd` | XAD | plain Rust, value only |
|---|---|---|---|---|---|---|
| Monte Carlo, price + 5 Greeks | 2.18 ms | 2.92 ms | 4.48 ms | 42.2 ms | 1.44 ms |
| Function, value + gradient | 1.62 ms | 2.22 ms | 2.11 ms | 25.5 ms | 1.59 ms |
| Function, gradient + H·v | 2.53 ms | 3.05 ms | | | |
| SABR vol, gradient + H·v | 24.8 ms | 58.7 ms | | | |

The raads value-only times are 1.20 ms (Monte Carlo) and 1.10 ms (function). Hand vectorisation speeds up Enzyme's value-only Monte Carlo (1.22 ms) but slows its reverse pass.

Versions: Enzyme from `nightly-2026-09-01`; XAD `main` 9c33936, Release, strong inlining, no thread-local tape, Apple clang 17 `-O3 -mcpu=native`. XAD's JIT backend needs a commercial licence and is not included.

### Run

```bash
benchmarks/run.sh   # fetches and builds XAD into target/
```

Needs the pinned nightly (`rust-toolchain.toml`) with the `enzyme` component (`rustup component add enzyme`), a C++17 compiler, git and cmake. Override them with `NIGHTLY=...`, `CMAKE="uvx cmake"` or `CXX=g++`. Everything is built with `-C target-cpu=native`.
