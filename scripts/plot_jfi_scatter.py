#!/usr/bin/env python3
"""
J_R vs J_I diagnostic scatter plot — bottleneck pathology diagnostic.

Plots jfi_resources (J_R) vs jfi_influence (J_I) for all solutions from:
  (a) a Pareto batch run  (metrics.csv from run_pareto_batch.py)
  (b) a scalar benchmark run  (results.csv from benchmark_engine.py)

Overlays all algorithms on the same axes so that dimensional collapse
in the min(J_R, J_I) bottleneck objective (if any) is visible: collapse
appears as clustering near the axes; correct bottleneck behavior produces
a dense upper-right quadrant cluster for all methods.

Budget alignment guard:
  Before plotting, both CSVs are inner-joined on seed.  The run_config.json
  of each input directory is inspected to detect budget mismatches.  If the
  Pareto run used a different number of iterations (generations × pop) than
  the scalar run used as its evaluation budget, the plot is a budget confound,
  not a bottleneck diagnostic.  The script aborts with a non-zero exit code
  unless --force-align is passed.

Usage:
    python scripts/plot_jfi_scatter.py \\
        --pareto-dir   output/run_20260314_120000 \\
        --benchmark-dir output/benchmark_20260314_120000 \\
        [--benchmark-budget 1000] \\
        [--force-align] \\
        [--output jfi_scatter.png]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J_R vs J_I bottleneck diagnostic scatter")
    p.add_argument("--pareto-dir",      required=True,
                   help="Output directory from run_pareto_batch.py (contains metrics.csv)")
    p.add_argument("--benchmark-dir",   required=True,
                   help="Output directory from benchmark_engine.py (contains results.csv)")
    p.add_argument("--benchmark-budget", type=int, default=None,
                   help="If benchmark results.csv covers multiple budgets, filter to this one")
    p.add_argument("--force-align",     action="store_true",
                   help="Proceed even if evaluation budgets differ across runs")
    p.add_argument("--collapse-threshold", type=float, default=0.05,
                   help="|J_R - J_I| > this value flags potential dimensional collapse (default 0.05)")
    p.add_argument("--output",          default=None,
                   help="Output PNG path (default: <pareto-dir>/jfi_scatter.png)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Budget alignment check
# ---------------------------------------------------------------------------

def _read_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def check_budget_alignment(pareto_cfg: dict, bench_cfg: dict, force: bool) -> None:
    """
    Compare evaluation effort across the two runs.

    run_pareto_batch uses `iterations` (Pareto generations) × pop_size.
    benchmark_engine uses `budget` (total evaluation count).
    We convert both to a common unit: total evaluations.
    """
    pareto_evals = None
    bench_evals  = None

    if "iterations" in pareto_cfg and "pop_size" in pareto_cfg:
        pareto_evals = pareto_cfg["iterations"] * pareto_cfg["pop_size"]
    if "budgets" in bench_cfg and bench_cfg["budgets"]:
        # budgets may be a comma-separated string or a list
        raw = bench_cfg["budgets"]
        if isinstance(raw, str):
            bench_evals = max(int(b) for b in raw.split(",") if b.strip())
        elif isinstance(raw, list):
            bench_evals = max(int(b) for b in raw)

    if pareto_evals is not None and bench_evals is not None:
        ratio = max(pareto_evals, bench_evals) / max(1, min(pareto_evals, bench_evals))
        if ratio > 5.0:
            msg = (
                f"Budget mismatch: Pareto run used ~{pareto_evals:,} evaluations "
                f"but benchmark run used ~{bench_evals:,} evaluations "
                f"(ratio {ratio:.1f}×). The scatter would be a budget confound, "
                f"not a bottleneck diagnostic. Pass --force-align to override."
            )
            if not force:
                print(f"ERROR: {msg}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"WARNING: {msg}", file=sys.stderr)
    else:
        # Could not determine budgets; warn but do not abort
        print(
            "WARNING: Could not determine evaluation budgets from run_config.json files. "
            "Ensure both runs used comparable evaluation effort.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pareto(pareto_dir: Path) -> pd.DataFrame:
    path = pareto_dir / "metrics.csv"
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"seed", "jfi_resources", "jfi_influence", "jains_index"}
    missing = required - set(df.columns)
    if missing:
        print(
            f"ERROR: metrics.csv is missing columns: {missing}. "
            "Re-run run_pareto_batch.py with the updated script.",
            file=sys.stderr,
        )
        sys.exit(1)
    df["algorithm"] = "NSGA-II"
    return df[["seed", "algorithm", "jfi_resources", "jfi_influence", "jains_index"]]


def load_benchmark(bench_dir: Path, budget_filter: int | None) -> pd.DataFrame:
    path = bench_dir / "results.csv"
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"seed", "algorithm", "jfi_resources", "jfi_influence", "jains_index"}
    missing = required - set(df.columns)
    if missing:
        print(
            f"ERROR: results.csv is missing columns: {missing}. "
            "Ensure benchmark_engine.py outputs these fields.",
            file=sys.stderr,
        )
        sys.exit(1)
    if budget_filter is not None and "budget" in df.columns:
        df = df[df["budget"] == budget_filter]
        if df.empty:
            print(
                f"ERROR: No rows in results.csv with budget={budget_filter}. "
                f"Available budgets: {sorted(pd.read_csv(path)['budget'].unique())}",
                file=sys.stderr,
            )
            sys.exit(1)
    # Keep only best solution per (seed, algorithm) = row with max jains_index
    df = (
        df.sort_values("jains_index", ascending=False)
          .groupby(["seed", "algorithm"], as_index=False)
          .first()
    )
    return df[["seed", "algorithm", "jfi_resources", "jfi_influence", "jains_index"]]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

MARKERS = {
    "NSGA-II": "o",
    "sa":      "s",
    "hc":      "^",
    "ts":      "D",
    "sga":     "P",
    "rs":      "X",
}
COLORS = {
    "NSGA-II": "#2196F3",
    "sa":      "#E91E63",
    "hc":      "#FF9800",
    "ts":      "#4CAF50",
    "sga":     "#9C27B0",
    "rs":      "#9E9E9E",
}


def plot_scatter(df: pd.DataFrame, output_path: Path, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    legend_handles = []
    for algo, group in df.groupby("algorithm"):
        color  = COLORS.get(algo, "#333333")
        marker = MARKERS.get(algo, "o")
        ax.scatter(
            group["jfi_resources"], group["jfi_influence"],
            c=color, marker=marker, alpha=0.55, s=28, linewidths=0.3,
            edgecolors="white", label=algo,
        )
        legend_handles.append(
            mpatches.Patch(color=color, label=algo)
        )

    # Diagonal J_R = J_I reference line
    lims = [0.0, 1.05]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="J_R = J_I")

    # Collapse fraction annotation per algorithm
    annotations = []
    for algo, group in df.groupby("algorithm"):
        collapse = (np.abs(group["jfi_resources"] - group["jfi_influence"]) > threshold).mean()
        annotations.append(f"{algo}: {collapse:.1%} with |J_R−J_I| > {threshold}")

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("$J_R$ (Resource JFI)", fontsize=12)
    ax.set_ylabel("$J_I$ (Influence JFI)", fontsize=12)
    ax.set_title("Bottleneck Pathology Diagnostic: $J_R$ vs $J_I$", fontsize=13)
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    # Annotation box
    annotation_text = "Collapse fractions (|J_R−J_I| > {:.2f}):\n".format(threshold)
    annotation_text += "\n".join(annotations)
    ax.text(
        0.02, 0.98, annotation_text,
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    pareto_dir = Path(args.pareto_dir)
    bench_dir  = Path(args.benchmark_dir)

    pareto_cfg = _read_config(pareto_dir)
    bench_cfg  = _read_config(bench_dir)

    check_budget_alignment(pareto_cfg, bench_cfg, args.force_align)

    pareto_df = load_pareto(pareto_dir)
    bench_df  = load_benchmark(bench_dir, args.benchmark_budget)

    # Inner-join on seed — only matched seeds contribute to the plot
    pareto_seeds = set(pareto_df["seed"])
    bench_seeds  = set(bench_df["seed"])
    common_seeds = pareto_seeds & bench_seeds

    n_pareto_only = len(pareto_seeds - common_seeds)
    n_bench_only  = len(bench_seeds  - common_seeds)
    if n_pareto_only or n_bench_only:
        print(
            f"Seed alignment: {len(common_seeds)} shared seeds; "
            f"{n_pareto_only} Pareto-only; {n_bench_only} benchmark-only. "
            "Only shared seeds are plotted.",
            file=sys.stderr,
        )

    if not common_seeds:
        print("ERROR: No seeds in common between the two inputs.", file=sys.stderr)
        return 1

    combined = pd.concat([
        pareto_df[pareto_df["seed"].isin(common_seeds)],
        bench_df[bench_df["seed"].isin(common_seeds)],
    ], ignore_index=True)

    output_path = Path(args.output) if args.output else pareto_dir / "jfi_scatter.png"
    plot_scatter(combined, output_path, args.collapse_threshold)

    # Also write seeds JSON for reproducibility
    seeds_json = output_path.with_suffix("").with_name(output_path.stem + "_seeds.json")
    with open(seeds_json, "w") as f:
        json.dump({"common_seeds": sorted(common_seeds)}, f, indent=2)
    print(f"Seed list: {seeds_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
