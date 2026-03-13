#!/usr/bin/env python3
"""
Existence proof figure: two maps with identical JFI but distinct spatial clustering.

Demonstrates that scalar fairness metrics are necessary but not sufficient on
topologically embedded problems — the paper's opening empirical claim (Figure 1).

Algorithm:
  1. Run SA on --search-seeds seeds, collect per-seed (jains_index, morans_i).
  2. Search for a pair (C₁, C₂) satisfying:
       |jains_index(C₁) − jains_index(C₂)| < --eps   (JFI-matched)
       |morans_i(C₁)   − morans_i(C₂)|   > --delta  (meaningfully different clustering)
  3. Regenerate both maps deterministically from their seeds.
  4. Render side-by-side hex-grid figures using plot_hex_map() from map_viz.py.
  5. Save jfi_matched_pair.png + jfi_matched_pair_seeds.json.

Exit codes:
  0 — pair found and figure saved
  1 — search exhausted without finding a valid pair (see message)
  2 — other error

The map generator is seeded and deterministic: generate_random_map(random_seed=S)
always produces the same starting map, so no layout storage is required.

Usage:
    python scripts/find_jfi_matched_pair.py \\
        [--search-seeds 500] [--eps 0.01] [--delta 0.15] \\
        [--sa-budget 1000] [--players 6] [--output-dir output/]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find two maps: JFI-matched but spatially distinct (existence proof)"
    )
    p.add_argument("--search-seeds", type=int,   default=500,
                   help="Number of seeds to search (default 500)")
    p.add_argument("--eps",          type=float, default=0.01,
                   help="|ΔJFI| < eps (default 0.01)")
    p.add_argument("--delta",        type=float, default=0.15,
                   help="|ΔMoran's I| > delta (default 0.15)")
    p.add_argument("--sa-budget",    type=int,   default=1000,
                   help="SA evaluation budget per seed (default 1000)")
    p.add_argument("--players",      type=int,   default=6,
                   help="Number of players (default 6)")
    p.add_argument("--base-seed",    type=int,   default=0,
                   help="Starting seed index (default 0)")
    p.add_argument("--sa-rate",      type=float, default=0.80)
    p.add_argument("--sa-min-temp",  type=float, default=0.01)
    p.add_argument("--output-dir",   type=str,   default="output",
                   help="Root output directory (default output/)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Search phase
# ---------------------------------------------------------------------------

def _run_sa_for_seed(seed, args, evaluator):
    """Run SA on one seed; return (ti4_map_optimized, jains_index, morans_i)."""
    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.spatial_optimizer import improve_balance_spatial

    ti4_map = generate_random_map(n_players=args.players, random_seed=seed)
    best_score, _history, _etb = improve_balance_spatial(
        ti4_map,
        evaluator,
        iterations=args.sa_budget,
        initial_acceptance_rate=args.sa_rate,
        min_temp=args.sa_min_temp,
        verbose=False,
    )
    return ti4_map, float(best_score.jains_index), float(best_score.morans_i)


def search_for_pair(args, evaluator):
    """
    Iterate over seeds until a JFI-matched but spatially-distinct pair is found.

    Returns:
        (seed_a, seed_b, results) where results is a list of per-seed dicts,
        or (None, None, results) if the search is exhausted.
    """
    results = []

    for i in range(args.search_seeds):
        seed = args.base_seed + i
        try:
            _map, jfi, mi = _run_sa_for_seed(seed, args, evaluator)
            results.append({"seed": seed, "jains_index": jfi, "morans_i": mi})
        except Exception as exc:
            print(f"  seed {seed}: skipped ({exc})", file=sys.stderr)
            continue

        if (i + 1) % 50 == 0:
            print(f"  Searched {i + 1}/{args.search_seeds} seeds …")

        # Check all previously collected seeds for a match
        for prev in results[:-1]:
            jfi_diff = abs(jfi - prev["jains_index"])
            mi_diff  = abs(mi  - prev["morans_i"])
            if jfi_diff < args.eps and mi_diff > args.delta:
                print(
                    f"  Pair found: seed {prev['seed']} (JFI={prev['jains_index']:.4f}, "
                    f"I={prev['morans_i']:+.4f}) vs seed {seed} "
                    f"(JFI={jfi:.4f}, I={mi:+.4f}) — "
                    f"|ΔJFI|={jfi_diff:.4f} < {args.eps}, "
                    f"|ΔI|={mi_diff:.4f} > {args.delta}"
                )
                return prev["seed"], seed, results

    return None, None, results


# ---------------------------------------------------------------------------
# Render phase
# ---------------------------------------------------------------------------

def render_pair(seed_a, seed_b, results_map, args, out_dir, evaluator):
    """Regenerate both maps from seeds and render side-by-side."""
    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.spatial_optimizer import improve_balance_spatial
    from ti4_analysis.visualization.map_viz import plot_hex_map

    maps_and_scores = []
    for seed in (seed_a, seed_b):
        ti4_map = generate_random_map(n_players=args.players, random_seed=seed)
        best_score, _history, _etb = improve_balance_spatial(
            ti4_map,
            evaluator,
            iterations=args.sa_budget,
            initial_acceptance_rate=args.sa_rate,
            min_temp=args.sa_min_temp,
            verbose=False,
        )
        maps_and_scores.append((ti4_map, best_score))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Existence Proof: Identical JFI, Distinct Spatial Clustering\n"
        "Caption: Scalar fairness metrics are blind to spatial structure.",
        fontsize=11,
    )

    for ax, (ti4_map, score), seed in zip(axes, maps_and_scores, (seed_a, seed_b)):
        plot_hex_map(ti4_map, ax=ax, show_system_ids=True)
        metric_text = (
            f"Seed {seed}\n"
            f"JFI = {score.jains_index:.4f}  "
            f"(R={score.jfi_resources:.3f}, I={score.jfi_influence:.3f})\n"
            f"Moran's I = {score.morans_i:+.4f}\n"
            f"LSAP = {score.lisa_penalty:.4f}"
        )
        ax.set_title(metric_text, fontsize=9, pad=4)

    fig.tight_layout()
    fig_path = out_dir / "jfi_matched_pair.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator
    evaluator = create_joebrew_evaluator()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Searching {args.search_seeds} seeds for JFI-matched pair "
        f"(|ΔJFI| < {args.eps}, |ΔI| > {args.delta}) …"
    )

    seed_a, seed_b, results = search_for_pair(args, evaluator)

    # Always save the search log
    search_log_path = out_dir / "jfi_pair_search_log.json"
    with open(search_log_path, "w") as f:
        json.dump(results, f, indent=2)

    if seed_a is None:
        msg = (
            f"Search exhausted: no JFI-matched pair found in {args.search_seeds} seeds "
            f"(|ΔJFI| < {args.eps}, |ΔI| > {args.delta}). "
            f"Try increasing --search-seeds, decreasing --eps, or decreasing --delta."
        )
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    # Render the figure
    render_pair(seed_a, seed_b, results, args, out_dir, evaluator)

    # Save seeds JSON for reproducibility
    seed_a_metrics = next(r for r in results if r["seed"] == seed_a)
    seed_b_metrics = next(r for r in results if r["seed"] == seed_b)
    seeds_record = {
        "seed_a":    seed_a_metrics,
        "seed_b":    seed_b_metrics,
        "jfi_diff":  round(abs(seed_a_metrics["jains_index"] - seed_b_metrics["jains_index"]), 6),
        "mi_diff":   round(abs(seed_a_metrics["morans_i"]    - seed_b_metrics["morans_i"]),    6),
        "eps":       args.eps,
        "delta":     args.delta,
        "sa_budget": args.sa_budget,
        "players":   args.players,
    }
    seeds_path = out_dir / "jfi_matched_pair_seeds.json"
    with open(seeds_path, "w") as f:
        json.dump(seeds_record, f, indent=2)
    print(f"Seeds:  {seeds_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
