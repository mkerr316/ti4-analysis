#!/usr/bin/env python3
"""
Publication-quality figures for benchmark statistical analysis.

Figures:
    fig1_composite_violin.pdf   — Composite score distributions (violin + swarm)
    fig2_per_objective.pdf      — Per-objective box plots (4-panel)
    fig3_convergence.pdf        — Anytime performance vs evaluation budget
    fig4_seed_heatmap.pdf       — Per-seed × algorithm composite score matrix
    fig5_effect_size_forest.pdf — VDA effect size forest plot with bootstrap CIs

Reads:
    <run_dir>/results.csv
    <run_dir>/stats/pairwise_tests.csv  (from analyze_benchmark.py)
    <run_dir>/stats/bootstrap_cis.csv

Usage:
    python scripts/plot_statistical_results.py output/benchmark_*/results.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

ALGO_NAMES = {"rs": "RS", "hc": "HC", "sa": "SA", "nsga2": "NSGA-II", "ts": "TS"}
ALGO_ORDER = ["rs", "hc", "sa", "nsga2", "ts"]
PALETTE = {
    "RS": "#937860", "HC": "#4C72B0", "SA": "#DD8452",
    "NSGA-II": "#55A868", "TS": "#C44E52",
}


def parse_args():
    p = argparse.ArgumentParser(description="Publication figures for benchmark results")
    p.add_argument("csv", type=Path, help="Path to results.csv")
    p.add_argument("--budget", type=int, default=None,
                   help="Filter to single budget (for single-budget figures)")
    p.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    return p.parse_args()


def _load(csv_path: Path, budget: int = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "budget" in df.columns and budget is not None:
        df = df[df["budget"] == budget]
    elif "budget" in df.columns and df["budget"].nunique() > 1 and budget is None:
        df = df[df["budget"] == df["budget"].min()]
    df["Algorithm"] = df["algorithm"].map(ALGO_NAMES)
    return df


# ---------------------------------------------------------------------------
# Figure 1: Composite score violin + swarm
# ---------------------------------------------------------------------------

def fig1_composite_violin(df: pd.DataFrame, out_dir: Path, fmt: str):
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.violinplot(
        data=df, x="Algorithm", y="composite_score",
        hue="Algorithm", legend=False,
        order=[ALGO_NAMES[a] for a in ALGO_ORDER],
        hue_order=[ALGO_NAMES[a] for a in ALGO_ORDER],
        palette=PALETTE, inner="box", linewidth=0.8,
        cut=0, ax=ax,
    )
    sns.stripplot(
        data=df, x="Algorithm", y="composite_score",
        order=[ALGO_NAMES[a] for a in ALGO_ORDER],
        color="0.25", size=2, alpha=0.3, jitter=True, ax=ax, legend=False,
    )

    ax.set_ylabel("Composite Score (lower = better)")
    ax.set_xlabel("")
    ax.set_title("Algorithm Comparison: Composite Score Distribution")

    # Annotate medians
    for i, algo in enumerate(ALGO_ORDER):
        med = df[df["algorithm"] == algo]["composite_score"].median()
        ax.annotate(
            f"med={med:.1f}", xy=(i, med), xytext=(0.3, 0),
            textcoords="offset fontsize", fontsize=8, color="0.3",
            va="center",
        )

    fig.savefig(out_dir / f"fig1_composite_violin.{fmt}")
    plt.close(fig)
    print(f"  fig1_composite_violin.{fmt}")


# ---------------------------------------------------------------------------
# Figure 2: Per-objective box plots (4-panel)
# ---------------------------------------------------------------------------

def fig2_per_objective(df: pd.DataFrame, out_dir: Path, fmt: str):
    metrics = [
        ("composite_score", "Composite Score", False),
        ("jains_index", "Jain's Fairness Index", True),
        ("morans_i", "|Moran's I|", False),
        ("lisa_penalty", "LISA Penalty", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    order = [ALGO_NAMES[a] for a in ALGO_ORDER]

    for ax, (metric, label, higher_better) in zip(axes.flat, metrics):
        plot_df = df.copy()
        if metric == "morans_i":
            plot_df[metric] = plot_df[metric].abs()

        sns.boxplot(
            data=plot_df, x="Algorithm", y=metric,
            hue="Algorithm", legend=False,
            order=order, hue_order=order,
            palette=PALETTE, linewidth=0.8,
            fliersize=2, ax=ax,
        )
        direction = "(higher = better)" if higher_better else "(lower = better)"
        ax.set_title(f"{label} {direction}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(label)

    fig.suptitle("Per-Objective Distributions", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig2_per_objective.{fmt}")
    plt.close(fig)
    print(f"  fig2_per_objective.{fmt}")


# ---------------------------------------------------------------------------
# Figure 3: Convergence profile (multi-budget)
# ---------------------------------------------------------------------------

def fig3_convergence(csv_path: Path, out_dir: Path, fmt: str):
    """Median composite vs budget. Requires multi-budget CSV."""
    df = pd.read_csv(csv_path)
    if "budget" not in df.columns or df["budget"].nunique() < 2:
        print("  fig3_convergence: SKIPPED (single budget, need --budgets run)")
        return

    df["Algorithm"] = df["algorithm"].map(ALGO_NAMES)
    budgets = sorted(df["budget"].unique())

    fig, ax = plt.subplots(figsize=(6, 4))

    for algo in ALGO_ORDER:
        name = ALGO_NAMES[algo]
        medians, q25s, q75s = [], [], []
        for b in budgets:
            vals = df[(df["algorithm"] == algo) & (df["budget"] == b)]["composite_score"]
            medians.append(vals.median())
            q25s.append(vals.quantile(0.25))
            q75s.append(vals.quantile(0.75))

        ax.plot(budgets, medians, "o-", label=name, color=PALETTE[name], linewidth=1.5)
        ax.fill_between(budgets, q25s, q75s, alpha=0.15, color=PALETTE[name])

    ax.set_xlabel("Evaluation Budget")
    ax.set_ylabel("Median Composite Score (lower = better)")
    ax.set_title("Anytime Performance Profile")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / f"fig3_convergence.{fmt}")
    plt.close(fig)
    print(f"  fig3_convergence.{fmt}")


# ---------------------------------------------------------------------------
# Figure 4: Per-seed heatmap
# ---------------------------------------------------------------------------

def fig4_seed_heatmap(df: pd.DataFrame, out_dir: Path, fmt: str):
    wide = df.pivot(index="seed", columns="algorithm", values="composite_score")
    algos = [a for a in ALGO_ORDER if a in wide.columns]
    wide = wide[algos].dropna()

    # Rank per seed (1=best)
    ranks = wide.rank(axis=1, method="min")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, max(4, len(wide) * 0.08)),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # Raw composite heatmap
    sns.heatmap(
        wide, ax=ax1, cmap="RdYlGn_r", linewidths=0.1,
        xticklabels=[ALGO_NAMES.get(a, a) for a in algos],
        yticklabels=True if len(wide) <= 50 else False,
        cbar_kws={"label": "Composite Score"},
    )
    ax1.set_title("Composite Score by Seed")
    ax1.set_ylabel("Seed")
    ax1.set_xlabel("")

    # Win-rate bar chart
    best_algo = wide.idxmin(axis=1)
    win_counts = best_algo.value_counts()
    colors = [PALETTE.get(ALGO_NAMES.get(a, a), "gray") for a in algos]
    bars = [win_counts.get(a, 0) for a in algos]
    ax2.barh([ALGO_NAMES.get(a, a) for a in algos], bars, color=colors)
    ax2.set_xlabel("Seeds Won (lowest composite)")
    ax2.set_title("Win Count")

    for i, v in enumerate(bars):
        ax2.text(v + 0.5, i, str(v), va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"fig4_seed_heatmap.{fmt}")
    plt.close(fig)
    print(f"  fig4_seed_heatmap.{fmt}")


# ---------------------------------------------------------------------------
# Figure 5: Effect size forest plot
# ---------------------------------------------------------------------------

def fig5_effect_forest(out_dir: Path, fmt: str):
    """Forest plot of VDA + bootstrap CIs. Requires stats/ outputs."""
    stats_dir = out_dir.parent / "stats"
    pw_path = stats_dir / "pairwise_tests.csv"
    boot_path = stats_dir / "bootstrap_cis.csv"

    if not pw_path.exists() or not boot_path.exists():
        print("  fig5_effect_forest: SKIPPED (run analyze_benchmark.py first)")
        return

    pw = pd.read_csv(pw_path)
    boot = pd.read_csv(boot_path)
    pw_comp = pw[pw["metric"] == "composite_score"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3),
                                    gridspec_kw={"width_ratios": [1, 1]})

    # Left panel: VDA
    pairs = pw_comp["pair"].tolist()
    vdas = pw_comp["vda_A"].values
    y_pos = np.arange(len(pairs))

    ax1.barh(y_pos, vdas - 0.5, left=0.5, height=0.5,
             color=["#DD8452" if v > 0.5 else "#4C72B0" for v in vdas], alpha=0.7)
    ax1.axvline(0.5, color="k", linewidth=0.8, linestyle="--")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(pairs)
    ax1.set_xlabel("Vargha-Delaney $\\hat{A}$")
    ax1.set_title("Effect Size (composite score)")
    ax1.set_xlim(0, 1)

    # Shade effect regions
    for bound in [0.44, 0.36, 0.29]:
        ax1.axvspan(bound, 1 - bound, alpha=0.03, color="gray")
    ax1.axvspan(0.44, 0.56, alpha=0.08, color="green", label="negligible")

    # Right panel: Bootstrap CI on median diff
    boot_pairs = boot["pair"].tolist()
    diffs = boot["median_diff"].values
    ci_lo = boot["ci_lower"].values
    ci_hi = boot["ci_upper"].values
    y_pos2 = np.arange(len(boot_pairs))

    ax2.errorbar(diffs, y_pos2, xerr=[diffs - ci_lo, ci_hi - diffs],
                 fmt="o", color="0.2", capsize=4, linewidth=1.2, markersize=5)
    ax2.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(boot_pairs)
    ax2.set_xlabel("Median Difference (composite score)")
    ax2.set_title("Bootstrap 95% CI")

    fig.tight_layout()
    fig.savefig(out_dir / f"fig5_effect_forest.{fmt}")
    plt.close(fig)
    print(f"  fig5_effect_forest.{fmt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found", file=sys.stderr)
        return 1

    df = _load(args.csv, args.budget)
    viz_dir = args.csv.parent / "viz"
    viz_dir.mkdir(exist_ok=True)

    print(f"Generating figures in {viz_dir}/")

    fig1_composite_violin(df, viz_dir, args.format)
    fig2_per_objective(df, viz_dir, args.format)
    fig3_convergence(args.csv, viz_dir, args.format)
    fig4_seed_heatmap(df, viz_dir, args.format)
    fig5_effect_forest(viz_dir, args.format)

    print(f"\nAll figures saved to {viz_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
