#!/usr/bin/env python3
# scripts/plot_results.py from auto-differentiation/ad-benchmarks (MIT, Copyright (c) 2026
# Xcelerit Computing Ltd.), with "raads" and "raads SIMD" added to the colours and order.
"""Generate benchmark comparison chart from a results CSV.

Usage:
    python scripts/plot_results.py [results.csv] [chart.png]

Defaults:
    results CSV: results/results.csv
    output:      results/chart.png  (also writes the same path with .svg)

Bar chart of gradient time per library, on a linear x-axis. A dashed
vertical reference line marks the "primal" baseline — the cost of one
evaluation of the same kernel with raw doubles, no AD machinery
attached. Bars to the left of the line are *below* primal cost (only
XAD-Codegen achieves this on the 8-input Heston benchmark, because
the JIT-compiled AVX2 graph is faster than the host C++ primal path).
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Visual config -------------------------------------------------------

XAD_GOLD = "#FBBA17"
XAD_GOLD_DARK = "#C28800"

LIBRARY_COLORS = {
    "FD":          "#CFD8DC",
    "autodiff":    "#B0BEC5",
    "CppAD":       "#9E9E9E",
    "Adept":       "#616161",
    "XAD":         XAD_GOLD,
    "XAD-Codegen": XAD_GOLD_DARK,
    "raads":       "#4C9BE8",
    "raads SIMD":  "#1F5FAD",
}

LIBRARY_ORDER = ["FD", "autodiff", "CppAD", "Adept", "XAD", "XAD-Codegen", "raads", "raads SIMD"]

# Libraries used to compute the canonical "primal" baseline. FD is excluded
# because its primal benchmark uses pre-generated random samples (whereas the
# AAD libs include RNG cost), so it understates the true single-eval cost.
PRIMAL_BASELINE_LIBS = ["XAD", "CppAD", "Adept", "XAD-Codegen"]

BENCH_LABELS = {
    "HestonMC":      "Heston MC\n8 sensitivities, 10K paths",
    "SABRCalib":     "SABR Calibration\n15 sensitivities, 500 iters",
    "XVA-CVA":       "XVA CVA\n40 sensitivities, 10K paths",
    "LiborSwaption": "LIBOR Swaption MC\n161 sensitivities, 10K paths",
}

BENCH_ORDER = ["HestonMC", "SABRCalib", "XVA-CVA", "LiborSwaption"]


# ---- Data loading --------------------------------------------------------

def load_results(csv_path):
    """Return {benchmark: {library: gradient_ms}}, primal baselines per
    benchmark, and the set of DNF entries."""
    data = defaultdict(dict)
    primals = defaultdict(dict)
    dnf = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grad = float(row["gradient_ms"])
            primal = float(row["primal_ms"])
            bench = row["benchmark"]
            lib = row["library"]
            if grad < 0:
                dnf.add((bench, lib))
            else:
                data[bench][lib] = grad
                if primal > 0:
                    primals[bench][lib] = primal
    # Median across the AAD baseline libs
    primal_baseline = {}
    for bench, libmap in primals.items():
        vals = sorted(libmap[l] for l in PRIMAL_BASELINE_LIBS if l in libmap)
        if vals:
            mid = len(vals) // 2
            primal_baseline[bench] = (vals[mid] if len(vals) % 2 == 1
                                       else 0.5 * (vals[mid - 1] + vals[mid]))
    return data, primal_baseline, dnf


def fmt_time(v):
    if v >= 1000:
        return f"{v / 1000:.2f} s"
    if v >= 100:
        return f"{v:.0f} ms"
    if v >= 10:
        return f"{v:.1f} ms"
    if v >= 1:
        return f"{v:.2f} ms"
    return f"{v:.3f} ms"


# ---- Plot ----------------------------------------------------------------

def plot_benchmarks(data, primal_baseline, dnf, output_path):
    benchmarks = [b for b in BENCH_ORDER if b in data]
    n = len(benchmarks)
    if n == 0:
        print("No benchmarks to plot.", file=sys.stderr)
        return

    # 2x2 grid for the 4-bench finance suite
    if n >= 5:
        rows, cols = 2, 3
    elif n >= 3:
        rows, cols = 2, 2
    else:
        rows, cols = 1, n

    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 2.9 * rows))
    fig.patch.set_facecolor("white")
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_idx, bench in enumerate(benchmarks):
        ax = axes[ax_idx]
        bench_data = data[bench]
        libs = [l for l in LIBRARY_ORDER if l in bench_data or (bench, l) in dnf]
        if not libs:
            ax.set_visible(False)
            continue

        positions = list(range(len(libs)))
        finite_positions = []
        finite_values = []
        finite_colors = []
        finite_labels = []
        dnf_positions = []
        for i, l in enumerate(libs):
            if (bench, l) in dnf:
                dnf_positions.append(i)
            else:
                finite_positions.append(i)
                finite_values.append(bench_data[l])
                finite_colors.append(LIBRARY_COLORS.get(l, "#CCCCCC"))
                finite_labels.append(fmt_time(bench_data[l]))

        bars = ax.barh(
            finite_positions, finite_values,
            color=finite_colors, edgecolor="white", linewidth=0.6, height=0.62,
        )

        ax.set_yticks(positions)
        ax.set_yticklabels(libs, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Gradient time (ms)", fontsize=8)
        ax.set_title(BENCH_LABELS.get(bench, bench),
                     fontsize=10, fontweight="bold", pad=8)

        max_val = max(finite_values) if finite_values else 1.0
        label_offset = max_val * 0.015
        for val, lbl, bar in zip(finite_values, finite_labels, bars):
            ax.text(bar.get_width() + label_offset,
                    bar.get_y() + bar.get_height() / 2,
                    lbl, va="center", ha="left", fontsize=7.8,
                    color="#222222")

        # Pad the x-axis so labels fit
        ax.set_xlim(left=0, right=max_val * 1.30)

        # Vertical primal-baseline reference line
        primal = primal_baseline.get(bench)
        if primal is not None and 0 < primal < max_val * 1.30:
            ax.axvline(primal, color="#444444", linestyle="--",
                       linewidth=1.0, alpha=0.7, zorder=0)
            ax.text(primal, -0.85,
                    f"primal {fmt_time(primal)}",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#444444", style="italic")

        # "DNF" labels
        for pos in dnf_positions:
            ax.text(max_val * 0.01, pos,
                    "DNF", va="center", ha="left", fontsize=8,
                    color="#999999", style="italic")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", which="major", alpha=0.25, linestyle=":")

    # Hide any unused subplot panels
    for j in range(len(benchmarks), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("AD library gradient time — lower is better. Dashed line: primal (no-AD) baseline.",
                 fontsize=11.5, fontweight="bold", y=1.00)

    plt.tight_layout(w_pad=2.5, h_pad=2.6, rect=(0, 0, 1, 0.96))

    base, _ = os.path.splitext(output_path)
    png_path = base + ".png"
    svg_path = base + ".svg"
    plt.savefig(png_path, bbox_inches="tight", dpi=160, facecolor="white")
    plt.savefig(svg_path, bbox_inches="tight", facecolor="white")
    print(f"Chart saved: {png_path}")
    print(f"Chart saved: {svg_path}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/chart.png"
    data, primal_baseline, dnf = load_results(csv_path)
    plot_benchmarks(data, primal_baseline, dnf, output_path)


if __name__ == "__main__":
    main()
