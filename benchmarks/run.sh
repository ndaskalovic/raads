#!/bin/sh
# Build and run raads, Enzyme and XAD on the same workloads. Needs the pinned nightly
# (rust-toolchain.toml) with the enzyme component, a C++17 compiler, git and cmake
# (override with CMAKE="uvx cmake", NIGHTLY=...).
set -e
cd "$(dirname "$0")/.."
CMAKE=${CMAKE:-cmake}
CXX=${CXX:-clang++}
case $(uname -m) in x86_64) CPU=-march=native ;; *) CPU=-mcpu=native ;; esac
XAD=target/xad

[ -d $XAD ] || git clone --depth 1 https://github.com/auto-differentiation/xad.git $XAD
[ -f $XAD/build/lib/libxad.a ] || {
    $CMAKE -S $XAD -B $XAD/build -DCMAKE_BUILD_TYPE=Release -DXAD_ENABLE_TESTS=OFF \
        -DXAD_USE_STRONG_INLINE=ON -DXAD_NO_THREADLOCAL=ON -DCMAKE_CXX_FLAGS="$CPU"
    $CMAKE --build $XAD/build -j 4
}
$CXX -std=c++17 -O3 $CPU -DNDEBUG -I $XAD/src -I $XAD/build/src benchmarks/xad/bench.cpp $XAD/build/lib/libxad.a -o target/xad_bench
RUSTFLAGS="-C target-cpu=native" cargo build --release -p raads-bench --bin raads
RUSTFLAGS="-C target-cpu=native -Zautodiff=Enable" cargo ${NIGHTLY:++$NIGHTLY} build --release -p raads-bench --features enzyme --bin enzyme --target-dir target/enzyme

target/release/raads
target/enzyme/release/enzyme
target/xad_bench
