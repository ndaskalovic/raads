# Benchmarks

## XAD's benchmark suite

![Benchmark chart](ad-benchmarks/chart.png)

The Heston MC, SABR calibration, XVA and LIBOR swaption workloads of [auto-differentiation/ad-benchmarks](https://github.com/auto-differentiation/ad-benchmarks), the suite in [XAD's README](https://github.com/auto-differentiation/xad#benchmarks), with raads added. Gradient time; Primal is the same workload in plain C++ doubles (the dashed line).

| # | Benchmark | Sensis | Primal | FD | XAD | Adept | autodiff | raads | raads SIMD |
|---|---|:-:|--:|--:|--:|--:|--:|--:|--:|
| 1 | Heston MC | 8 | 5.16 ms | 46.5 ms | 44.9 ms | 68.6 ms | DNF | 17.0 ms | **2.54 ms** |
| 2 | SABR Calib | 15 | 0.851 ms | 13.5 ms | 6.81 ms | 13.6 ms | 17.3 ms | 0.930 ms | **0.777 ms** |
| 3 | XVA CVA | 40 | 237 ms | 9.74 s | 1.62 s | 5.26 s | DNF | 877 ms | **123 ms** |
| 4 | LIBOR Swaption | 161 | 49.9 ms | 7.52 s | 830 ms | 915 ms | DNF | 129 ms | **55.3 ms** |

- Same inputs, sizes and timing as the suite: its own code generates the random numbers and market data ([`ad-benchmarks/reference.cpp`](ad-benchmarks/reference.cpp)), median of 10 runs after 3 warm-ups, f64, single core. raads's values and gradients match XAD's to 1e-9, except two XVA gradients (below).
- raads runs the suite's kernels ported line by line ([`src/bin/adbench.rs`](src/bin/adbench.rs)): Heston as a path model, the SABR objective as a kernel over strikes, LIBOR and XVA as one kernel per path with their loops, XVA's curve lookups (interval and weight of a time) in plain Rust. raads: 1 lane; raads SIMD: 8 lanes (Heston), 4 (SABR, LIBOR), 16 (XVA).
- XVA's derivatives w.r.t. `rates[10]` and `vols[0]` are left out of the check: they are below f64 resolution. A few paths reach CVAs up to 4e44 (the mean is 3e41) and dominate these two means. There raads's reverse mode, forward mode (`Dual`) and XAD disagree with each other by 10–50%, and on one such path 80-digit arithmetic gives a `rates[10]` derivative five orders of magnitude smaller than f64 does.
- The suite's LIBOR timings include drawing the normals inside the timed loop; raads reads them pre-drawn. Drawing them costs 5.1 ms, under 1% of XAD's time.
- Apple M5, Apple clang 17. Differences from the suite's published run (Xeon, GCC 13, `-O3 -mavx2 -mfma`): `-mcpu=native` on ARM; CppAD left out (its build needs pkg-config); XAD-Codegen is commercial.
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
