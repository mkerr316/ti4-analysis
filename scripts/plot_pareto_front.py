#!/usr/bin/env python3
"""
plot_pareto_front.py

Extracts and visualizes the global Pareto front from a batch of multi-objective TI4 map evaluations.
Minimizes: Balance Gap, |Moran's I|, and (1 - Jain's Index).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def extract_pareto_front(costs: np.ndarray) -> np.ndarray:
    """
    Memory-safe, fast iterative extraction of the Pareto frontier.
    Returns a boolean mask of the efficient points.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point that is strictly better than 'c' in at least one dimension
            # Points identical to 'c' will evaluate to False and be dropped (keeping only one unique)
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Always keep self during evaluation
    return is_efficient

def main():
    parser = argparse.ArgumentParser(description="Visualize Global Pareto Front")
    parser.add_argument("--csv", type=str, required=True, help="Path to metrics.csv")
    parser.add_argument("--out", type=str, help="Output directory for plots (defaults to csv dir)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out) if args.out else csv_path.parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(csv_path)
    df = df[df['solution_rank'] >= 0].copy()  # Drop errors

    # 2. Build Objective Matrix M (Minimization)
    df['abs_morans_i'] = df['morans_i'].abs()
    df['jains_inv'] = 1.0 - df['jains_index']

    M = df[['balance_gap', 'abs_morans_i', 'jains_inv']].values

    # Extract Global Pareto Front
    pareto_mask = extract_pareto_front(M)
    front_df = df[pareto_mask].copy()

    # Save global front CSV
    front_df.drop(columns=['abs_morans_i', 'jains_inv']).to_csv(out_dir / "global_pareto_front.csv", index=False)

    # Stdout Summary
    print(f"Total solutions evaluated: {len(df)}")
    print(f"Global Pareto front size:  {len(front_df)}")

    best_gap = front_df.loc[front_df['balance_gap'].idxmin()]
    best_moran = front_df.loc[front_df['abs_morans_i'].idxmin()]
    best_jain = front_df.loc[front_df['jains_index'].idxmax()]

    print(f"Best Balance Gap:  {best_gap['balance_gap']:.1f} (Seed {int(best_gap['seed'])}, Rank {int(best_gap['solution_rank'])})")
    print(f"Best |Moran's I|:  {best_moran['abs_morans_i']:.4f} (Seed {int(best_moran['seed'])}, Rank {int(best_moran['solution_rank'])})")
    print(f"Best Jain's Index: {best_jain['jains_index']:.4f} (Seed {int(best_jain['seed'])}, Rank {int(best_jain['solution_rank'])})")

    # Setup plotting styles
    plt.style.use('seaborn-v0_8-darkgrid')

    # --- Figure 1: Balance Gap vs Moran's I ---
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.scatter(df['balance_gap'], df['abs_morans_i'], c='grey', alpha=0.15, s=4, label='All Solutions')

    scatter1 = ax1.scatter(
        front_df['balance_gap'], front_df['abs_morans_i'],
        c=front_df['jains_index'], cmap='viridis', s=60,
        edgecolor='k', zorder=5, label='Global Pareto Front'
    )

    # Annotate perfect gap if it exists
    perfect_gap = front_df[front_df['balance_gap'] == 0]
    if not perfect_gap.empty:
        p = perfect_gap.iloc[0]
        ax1.annotate(f"Seed {int(p['seed'])} (Gap 0)",
                     (p['balance_gap'], p['abs_morans_i']),
                     xytext=(10, 10), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    cbar1 = fig1.colorbar(scatter1, ax=ax1)
    cbar1.set_label("Jain's Fairness Index")
    ax1.set_xlabel("Balance Gap (lower = more equal)")
    ax1.set_ylabel("Moran's I Magnitude: |value| (lower = less clustered)")
    ax1.set_title("Global Pareto Front: Balance Gap vs. Spatial Clustering")
    ax1.legend()

    fig1.savefig(out_dir / "fig1_gap_vs_morans.png", dpi=300, bbox_inches='tight')
    fig1.savefig(out_dir / "fig1_gap_vs_morans.svg", bbox_inches='tight')
    plt.close(fig1)

    # --- Figure 2: Balance Gap vs Jain's Index ---
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.scatter(df['balance_gap'], df['jains_index'], c='grey', alpha=0.15, s=4, label='All Solutions')

    scatter2 = ax2.scatter(
        front_df['balance_gap'], front_df['jains_index'],
        c=front_df['abs_morans_i'], cmap='plasma', s=60,
        edgecolor='k', zorder=5, label='Global Pareto Front'
    )

    cbar2 = fig2.colorbar(scatter2, ax=ax2)
    cbar2.set_label("|Moran's I|")
    ax2.set_xlabel("Balance Gap (lower = more equal)")
    ax2.set_ylabel("Jain's Fairness Index (higher = more fair)")
    ax2.set_title("Global Pareto Front: Balance Gap vs. Fairness")
    ax2.legend()

    fig2.savefig(out_dir / "fig2_gap_vs_jains.png", dpi=300, bbox_inches='tight')
    fig2.savefig(out_dir / "fig2_gap_vs_jains.svg", bbox_inches='tight')
    plt.close(fig2)

    # --- Figure 3: Marginal Distributions ---
    fig3, axes = plt.subplots(3, 1, figsize=(8, 12))
    metrics = [
        ('balance_gap', 'Balance Gap', 50),
        ('morans_i', "Moran's I (Raw)", 50),
        ('jains_index', "Jain's Fairness Index", 50)
    ]

    for ax, (col, title, bins) in zip(axes, metrics):
        ax.hist(df[col], bins=bins, color='grey', alpha=0.4, label='All Solutions', density=True)
        ax.hist(front_df[col], bins=bins//2, color='#1f77b4', alpha=0.8, label='Global Pareto Front', density=True)
        if col == 'morans_i':
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_title(title)
        ax.legend()

    fig3.suptitle("Metric Distributions — Global Pareto Front vs. All Solutions", y=0.92)
    fig3.savefig(out_dir / "fig3_marginals.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # --- Figure 4: Parallel Coordinates ---
    # Min-max scaling so lines don't flatten out
    M_front = front_df[['balance_gap', 'abs_morans_i', 'jains_inv']].values
    M_min, M_max = M_front.min(axis=0), M_front.max(axis=0)
    # Avoid div by zero if a dimension is perfectly flat
    range_span = np.where((M_max - M_min) == 0, 1e-9, M_max - M_min)
    M_norm = (M_front - M_min) / range_span

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('coolwarm')
    scores = front_df['composite_score'].values
    norm = Normalize(vmin=scores.min(), vmax=scores.max())

    x_coords = [0, 1, 2]
    for i in range(len(M_norm)):
        ax4.plot(x_coords, M_norm[i], color=cmap(norm(scores[i])), alpha=0.7, linewidth=2)

    ax4.set_xticks(x_coords)
    ax4.set_xticklabels([
        f"Balance Gap\n[{M_min[0]:.1f} - {M_max[0]:.1f}]",
        f"|Moran's I|\n[{M_min[1]:.4f} - {M_max[1]:.4f}]",
        f"1 - Jain's\n[{M_min[2]:.4f} - {M_max[2]:.4f}]"
    ])

    # Add colorbar for composite score
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar4 = fig4.colorbar(sm, ax=ax4)
    cbar4.set_label("Composite Score")
    ax4.set_title("Parallel Coordinates: Global Pareto Front Trade-offs (Min-Max Scaled)")
    ax4.set_ylabel("Normalized Range [0 = Best, 1 = Worst]")

    fig4.savefig(out_dir / "fig4_parallel_coords.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)

    print(f"\nAll visual artifacts written to: {out_dir}")

if __name__ == "__main__":
    main()
