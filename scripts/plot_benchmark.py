#!/usr/bin/env python3
"""
plot_benchmark.py

Reads benchmark results.csv and produces four publication-ready figures:
  Fig 1 — Composite score boxplot by algorithm (headline result)
  Fig 2 — Per-objective boxplots (balance_gap, lisa_penalty, |morans_i|)
  Fig 3 — Efficiency scatter (elapsed_sec vs composite_score)
  Fig 4 — Per-seed improvement over HC baseline (sorted by HC difficulty)

Usage:
    python scripts/plot_benchmark.py --csv output/sapelo2-run-20260310/results.csv
    python scripts/plot_benchmark.py --csv results.csv --out plots/ --dpi 150
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import chi2


# ── Constants ──────────────────────────────────────────────────────────────────

ALGO_ORDER  = ["hc", "sa", "nsga2"]
ALGO_LABELS = {"hc": "Greedy HC", "sa": "SA", "nsga2": "NSGA-II"}
PALETTE     = {"hc": "#4878CF", "sa": "#6ACC65", "nsga2": "#D65F5F"}

DISPLAY_ORDER  = [ALGO_LABELS[a] for a in ALGO_ORDER]
DISPLAY_PALETTE = {ALGO_LABELS[k]: v for k, v in PALETTE.items()}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_and_pivot(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_long  — long-form DataFrame with display-label 'algorithm' column
        df_wide  — wide-form pivoted on seed (one column per algo per metric)
    """
    df = pd.read_csv(csv_path)

    required = {"seed", "algorithm", "balance_gap", "morans_i", "jains_index",
                "lisa_penalty", "composite_score", "elapsed_sec"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: results.csv is missing columns: {missing}")

    df["abs_morans_i"] = df["morans_i"].abs()
    df["algorithm"] = df["algorithm"].map(ALGO_LABELS).fillna(df["algorithm"])

    # Wide pivot for seed-aligned per-seed comparisons
    df_wide = df.pivot(index="seed", columns="algorithm", values="composite_score").reset_index()

    return df, df_wide


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.png / .svg")


def _confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Draw a covariance ellipse for a 2-D cloud of points."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    # chi2 critical value for 2 DOF at 95 %
    scale = np.sqrt(chi2.ppf(0.95, df=2))
    angle = np.degrees(np.arctan2(*vecs[:, -1][::-1]))
    width, height = 2 * scale * np.sqrt(np.abs(vals))
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height,
        angle=angle, **kwargs
    )
    ax.add_patch(ell)


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def fig1_composite_boxplot(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.boxplot(
        data=df, x="algorithm", y="composite_score", hue="algorithm",
        order=DISPLAY_ORDER, palette=DISPLAY_PALETTE,
        width=0.5, linewidth=1.4, fliersize=0, legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df, x="algorithm", y="composite_score", hue="algorithm",
        order=DISPLAY_ORDER, palette=DISPLAY_PALETTE,
        size=3, alpha=0.3, jitter=True, dodge=False, legend=False,
        ax=ax,
    )

    # Mean markers
    means = df.groupby("algorithm")["composite_score"].mean()
    for i, algo in enumerate(DISPLAY_ORDER):
        ax.scatter(i, means[algo], marker="D", s=60,
                   color="white", edgecolor="black", linewidth=1.2, zorder=5)

    ax.set_xlabel("Algorithm", fontsize=13)
    ax.set_ylabel("Composite Score (lower = better)", fontsize=13)
    ax.set_title("Map Quality by Algorithm — 100 Random Seeds, Equal 1 000-Eval Budget", fontsize=13)
    ax.annotate("(D) = mean", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, color="grey")

    fig.tight_layout()
    _save(fig, out_dir, "fig1_composite_score_boxplot", dpi)


# ── Figure 2 ──────────────────────────────────────────────────────────────────

def fig2_per_objective(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    metrics = [
        ("balance_gap",  "Balance Gap\n(resource units, lower = more equal)"),
        ("lisa_penalty", "LISA Penalty\n(spatial autocorrelation, lower = better)"),
        ("abs_morans_i", "|Moran's I|\n(global clustering, lower = random)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)

    for ax, (col, ylabel) in zip(axes, metrics):
        sns.boxplot(
            data=df, x="algorithm", y=col, hue="algorithm",
            order=DISPLAY_ORDER, palette=DISPLAY_PALETTE,
            width=0.55, linewidth=1.3, fliersize=2, legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(col.replace("_", " ").title(), fontsize=12)
        ax.tick_params(axis="x", labelsize=11)

    fig.suptitle("Per-Objective Distributions by Algorithm", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_per_objective_boxplots", dpi)


# ── Figure 3 ──────────────────────────────────────────────────────────────────

def fig3_efficiency_scatter(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for algo in DISPLAY_ORDER:
        sub = df[df["algorithm"] == algo]
        color = DISPLAY_PALETTE[algo]
        ax.scatter(sub["elapsed_sec"], sub["composite_score"],
                   color=color, alpha=0.4, s=22, label=algo, zorder=3)
        _confidence_ellipse(
            sub["elapsed_sec"].values, sub["composite_score"].values,
            ax, n_std=2.0,
            edgecolor=color, facecolor=color, alpha=0.10, linewidth=1.5, zorder=2
        )
        # Mean crosshair
        ax.scatter(sub["elapsed_sec"].mean(), sub["composite_score"].mean(),
                   color=color, s=120, marker="D",
                   edgecolor="black", linewidth=1.0, zorder=5)

    ax.set_xlabel("Wall-clock Time (seconds)", fontsize=13)
    ax.set_ylabel("Composite Score (lower = better)", fontsize=13)
    ax.set_title("Quality vs. Speed — SA is Pareto-Dominant", fontsize=13)
    ax.legend(title="Algorithm", fontsize=11)
    ax.annotate("(D) = mean  /  ellipse = 95 % CI",
                xy=(0.98, 0.98), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="grey")

    fig.tight_layout()
    _save(fig, out_dir, "fig3_efficiency_scatter", dpi)


# ── Figure 4 ──────────────────────────────────────────────────────────────────

def fig4_per_seed_improvement(df_wide: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    hc_col    = ALGO_LABELS["hc"]
    sa_col    = ALGO_LABELS["sa"]
    nsga2_col = ALGO_LABELS["nsga2"]

    # Only keep seeds where all three algorithms have data
    complete = df_wide.dropna(subset=[hc_col, sa_col, nsga2_col]).copy()
    complete = complete.sort_values(hc_col).reset_index(drop=True)
    complete["rank"] = range(len(complete))

    sa_pct    = (complete[hc_col] - complete[sa_col])    / complete[hc_col] * 100
    nsga2_pct = (complete[hc_col] - complete[nsga2_col]) / complete[hc_col] * 100

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6, label="HC baseline (0 %)")
    ax.fill_between(complete["rank"], sa_pct, 0,
                    where=(sa_pct > 0), alpha=0.12, color=DISPLAY_PALETTE[sa_col])
    ax.fill_between(complete["rank"], sa_pct, 0,
                    where=(sa_pct <= 0), alpha=0.12, color="red")

    ax.plot(complete["rank"], nsga2_pct, color=DISPLAY_PALETTE[nsga2_col],
            linewidth=1.2, alpha=0.7, label=f"NSGA-II (mean {nsga2_pct.mean():.1f} %)")
    ax.plot(complete["rank"], sa_pct,    color=DISPLAY_PALETTE[sa_col],
            linewidth=1.6, label=f"SA (mean {sa_pct.mean():.1f} %)")

    ax.set_xlabel("Seed rank (sorted by HC difficulty, easy → hard)", fontsize=12)
    ax.set_ylabel("% improvement over HC\n(positive = better than HC)", fontsize=12)
    ax.set_title("Per-Seed Composite Score Improvement vs. Greedy HC Baseline", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, len(complete) - 1)

    fig.tight_layout()
    _save(fig, out_dir, "fig4_per_seed_improvement", dpi)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TI4 benchmark results")
    parser.add_argument("--csv",   required=True, type=Path, help="Path to results.csv")
    parser.add_argument("--out",   type=Path, default=None,  help="Output directory (default: <csv_dir>/viz/)")
    parser.add_argument("--dpi",   type=int,  default=300,   help="Figure DPI (default: 300)")
    parser.add_argument("--style", type=str,  default="seaborn-v0_8-whitegrid", help="Matplotlib style")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"ERROR: file not found: {args.csv}")

    out_dir = args.out or (args.csv.parent / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use(args.style)
    except OSError:
        print(f"WARNING: style '{args.style}' not found, using default.")

    print(f"Loading {args.csv} …")
    df_long, df_wide = load_and_pivot(args.csv)
    print(f"  {len(df_long)} rows, {df_long['algorithm'].nunique()} algorithms, "
          f"{df_long['seed'].nunique()} seeds\n")

    print("Generating figures …")
    fig1_composite_boxplot(df_long, out_dir, args.dpi)
    fig2_per_objective(df_long, out_dir, args.dpi)
    fig3_efficiency_scatter(df_long, out_dir, args.dpi)
    fig4_per_seed_improvement(df_wide, out_dir, args.dpi)

    print(f"\nDone. All figures written to: {out_dir.resolve()}")

    # Quick text summary
    print("\n── Algorithm Summary ──────────────────────────────────────────────")
    summary = (
        df_long
        .groupby("algorithm")["composite_score"]
        .agg(["mean", "std", "min", "max"])
        .reindex(DISPLAY_ORDER)
        .rename(columns={"mean": "Mean", "std": "Stdev", "min": "Min", "max": "Max"})
    )
    print(summary.to_string(float_format="{:.2f}".format))


if __name__ == "__main__":
    main()
