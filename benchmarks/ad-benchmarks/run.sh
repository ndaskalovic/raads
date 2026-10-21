#!/bin/sh
# Heston MC, SABR calibration and LIBOR swaption from auto-differentiation/ad-benchmarks (the
# suite in XAD's README), with raads added: builds and runs the suite, generates its inputs and
# XAD's results for raads (reference.cpp), runs raads (src/bin/adbench.rs) and draws chart.png
# (plot.py).
# Needs git, cmake (or CMAKE="uvx cmake"), a C++17 compiler, uv and the pinned nightly.
set -e
cd "$(dirname "$0")/../.."
CMAKE=${CMAKE:-cmake}
CXX=${CXX:-clang++}
SUITE=target/ad-benchmarks
case $(uname -m) in x86_64) CPU="-mavx2 -mfma" ;; *) CPU=-mcpu=native ;; esac

[ -d $SUITE ] || git clone -q https://github.com/auto-differentiation/ad-benchmarks.git $SUITE
git -C $SUITE checkout -q c8439f9e4dada0bfa32430d7fbfd902c083f72ac
# the suite's flags are x86-only; CppAD's default branch is now main
perl -0pi -e "s/-mavx2 -mfma/$CPU/; s|(CppAD.git\s+GIT_TAG\s+)master|\${1}main|" $SUITE/CMakeLists.txt
# CppAD's build needs pkg-config: left out
$CMAKE -S $SUITE -B $SUITE/build -DCMAKE_BUILD_TYPE=Release -DENABLE_CPPAD=OFF
$CMAKE --build $SUITE/build -j 4
(cd $SUITE && ./build/ad_benchmarks --only heston,sabr,libor)

# inputs and XAD results for raads, with the suite's own headers and XAD build
$CXX -std=c++17 -O3 $CPU -DNDEBUG -I $SUITE -I $SUITE/build/_deps/xad-src/src -I $SUITE/build/_deps/xad-build/src \
    benchmarks/ad-benchmarks/reference.cpp $SUITE/build/lib/libxad.a -o target/adbench_reference
target/adbench_reference > target/adbench_reference.txt
RUSTFLAGS="-C target-cpu=native" cargo build --release -p raads-bench --bin adbench
target/release/adbench

cat $SUITE/results/results.csv target/adbench_raads.csv > target/adbench_results.csv
uv run --with matplotlib python benchmarks/ad-benchmarks/plot.py target/adbench_results.csv benchmarks/ad-benchmarks/chart.png
