#!/usr/bin/env python3
"""
LSAP threshold sensitivity test (V4 reviewer rebuttal).

Evaluates whether Kendall's τ rank correlation between the baseline LSAP
(sum of all positive local Iᵢ) and the thresholded LSAP (Σ max(0, Iᵢ − τ))
is high enough to conclude the threshold is immaterial.

Pre-registered decision rule (committed before Sapelo2 run):
  τ_kendall > 0.90  → threshold immaterial; baseline LSAP is defended.
  0.80 ≤ τ_kendall ≤ 0.90 → marginal divergence; report both, note limitation.
  τ_kendall < 0.80  → rankings structurally sensitive to threshold;
                       requires theoretical justification for baseline choice,
                       or switch to thresholded variant as default.
This rule must be committed before observing the data.

Mathematical note on τ = 0.05:
  Under H₀ of spatial randomness, E[Iᵢ] = −1/(n−1).
  For n = 37 swappable tiles: E[Iᵢ] ≈ −0.028.
  τ = 0.05 is therefore defensible without Var[Iᵢ]:
    τ > |E[Iᵢ]| ≈ 0.028, placing the noise floor above the analytical
    null magnitude.  τ is an empirical noise-floor heuristic, NOT the
    formal statistical expectation under H₀.  Var[Iᵢ] under H₀ depends
    on the specific weight-matrix structure and cannot be derived
    analytically without permutation testing on this topology.

Usage:
    python scripts/lsap_threshold_sensitivity.py \\
        [--seeds 50] [--budget 1000] [--tau 0.05] [--output-dir output/]
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau


# ---------------------------------------------------------------------------
# Decision rule (pre-registered — do not change after running the experiment)
# ---------------------------------------------------------------------------

DECISION_RULE = {
    "defend_threshold":  0.90,   # tau > this → baseline LSAP is defended
    "marginal_threshold": 0.80,  # 0.80–0.90 → marginal divergence
    # tau < 0.80 → structurally sensitive; switch to thresholded or justify
}


def apply_decision_rule(tau: float) -> str:
    if tau > DECISION_RULE["defend_threshold"]:
        return (
            f"DEFENDED: τ_kendall = {tau:.4f} > {DECISION_RULE['defend_threshold']}. "
            "Threshold is immaterial; baseline LSAP is defended."
        )
    elif tau >= DECISION_RULE["marginal_threshold"]:
        return (
            f"MARGINAL: τ_kendall = {tau:.4f} in "
            f"[{DECISION_RULE['marginal_threshold']}, {DECISION_RULE['defend_threshold']}]. "
            "Marginal divergence — report both formulations and note as a limitation."
        )
    else:
        return (
            f"SENSITIVE: τ_kendall = {tau:.4f} < {DECISION_RULE['marginal_threshold']}. "
            "Rankings are structurally sensitive to the threshold choice. "
            "Theoretical justification required, or switch to thresholded LSAP as default."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LSAP threshold sensitivity test (V4 rebuttal)"
    )
    p.add_argument("--seeds",      type=int,   default=50,      help="Number of random seeds")
    p.add_argument("--budget",     type=int,   default=1000,    help="SA evaluation budget per seed")
    p.add_argument("--tau",        type=float, default=0.05,    help="Noise-floor threshold (default 0.05)")
    p.add_argument("--base-seed",  type=int,   default=0,       help="First random seed")
    p.add_argument("--players",    type=int,   default=6,       help="Number of players")
    p.add_argument("--sa-rate",    type=float, default=0.80,    help="SA initial acceptance rate")
    p.add_argument("--sa-min-temp",type=float, default=0.01,    help="SA min temperature")
    p.add_argument("--output-dir", type=str,   default="output",help="Root output directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core sensitivity evaluation
# ---------------------------------------------------------------------------

def run_sensitivity(args) -> List[dict]:
    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.spatial_optimizer import improve_balance_spatial
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

    evaluator = create_joebrew_evaluator()
    rows = []

    for i in range(args.seeds):
        seed = args.base_seed + i
        try:
            ti4_map = generate_random_map(n_players=args.players, random_seed=seed)
            t0 = time.time()
            best_map, best_score = improve_balance_spatial(
                ti4_map,
                evaluator,
                max_iterations=args.budget,
                initial_acceptance_rate=args.sa_rate,
                min_temp=args.sa_min_temp,
            )
            elapsed = time.time() - t0

            topology = MapTopology(best_map)
            fast = FastMapState(best_map, evaluator, topology=topology)

            baseline_lsap    = fast.lisa_penalty()
            thresholded_lsap = fast.lisa_penalty_thresholded(tau=args.tau)

            rows.append({
                "seed":            seed,
                "algorithm":       "sa",
                "budget":          args.budget,
                "baseline_lsap":   round(baseline_lsap,    6),
                "thresholded_lsap":round(thresholded_lsap, 6),
                "jains_index":     round(float(best_score.jains_index), 4),
                "morans_i":        round(float(best_score.morans_i),    4),
                "composite_score": round(float(best_score.composite_score()), 4),
                "elapsed_sec":     round(elapsed, 2),
            })

            if (i + 1) % 10 == 0:
                print(f"  seed {seed}: baseline={baseline_lsap:.4f}  "
                      f"thresh={thresholded_lsap:.4f}")

        except Exception as exc:
            print(f"  seed {seed}: ERROR — {exc}", file=sys.stderr)

    return rows


# ---------------------------------------------------------------------------
# Analysis and reporting
# ---------------------------------------------------------------------------

def analyze_and_report(rows: List[dict], tau: float, out_dir: Path) -> None:
    baseline    = np.array([r["baseline_lsap"]    for r in rows])
    thresholded = np.array([r["thresholded_lsap"] for r in rows])

    # Rank-based Kendall's τ (rankings, not raw values)
    baseline_rank    = baseline.argsort().argsort().astype(float)
    thresholded_rank = thresholded.argsort().argsort().astype(float)
    tau_stat, p_value = kendalltau(baseline_rank, thresholded_rank)

    verdict = apply_decision_rule(tau_stat)

    # Text report
    report_lines = [
        "LSAP Threshold Sensitivity Report",
        "==================================",
        f"Seeds: {len(rows)}",
        f"SA budget: {rows[0]['budget'] if rows else 'n/a'} evaluations",
        f"Threshold τ: {tau}",
        f"  (τ > |E[Iᵢ]| ≈ 0.028 for n=37 — empirical noise-floor heuristic)",
        "",
        f"Kendall's τ (rank correlation, baseline vs thresholded): {tau_stat:.4f}",
        f"p-value: {p_value:.4e}",
        "",
        "Pre-registered decision rule outcome:",
        f"  {verdict}",
        "",
        "Raw value statistics:",
        f"  Baseline LSAP    — mean: {baseline.mean():.4f}  std: {baseline.std():.4f}",
        f"  Thresholded LSAP — mean: {thresholded.mean():.4f}  std: {thresholded.std():.4f}",
        "",
        "Decision rule thresholds (pre-registered):",
        f"  > {DECISION_RULE['defend_threshold']} → baseline defended",
        f"  {DECISION_RULE['marginal_threshold']}–{DECISION_RULE['defend_threshold']} → marginal",
        f"  < {DECISION_RULE['marginal_threshold']} → sensitive",
    ]
    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = out_dir / "lsap_threshold_report.txt"
    report_path.write_text(report_text)
    print(f"\nReport: {report_path}")

    # Scatter plot: baseline vs thresholded raw values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: raw value scatter
    ax = axes[0]
    ax.scatter(baseline, thresholded, alpha=0.5, s=25, color="#E91E63")
    mn, mx = min(baseline.min(), thresholded.min()), max(baseline.max(), thresholded.max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel("Baseline LSAP", fontsize=11)
    ax.set_ylabel(f"Thresholded LSAP (τ={tau})", fontsize=11)
    ax.set_title("Raw LSAP Values", fontsize=12)
    ax.legend(fontsize=9)

    # Right: rank scatter
    ax2 = axes[1]
    ax2.scatter(baseline_rank, thresholded_rank, alpha=0.5, s=25, color="#2196F3")
    ax2.plot(
        [0, len(rows)], [0, len(rows)], "k--", linewidth=0.8, alpha=0.5, label="y = x"
    )
    ax2.set_xlabel("Baseline LSAP rank", fontsize=11)
    ax2.set_ylabel("Thresholded LSAP rank", fontsize=11)
    ax2.set_title(
        f"Rank Correlation (Kendall's τ = {tau_stat:.3f}, p = {p_value:.2e})",
        fontsize=12,
    )
    ax2.legend(fontsize=9)

    fig.suptitle(
        f"LSAP Threshold Sensitivity (τ = {tau})\n{verdict}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    plot_path = out_dir / "lsap_threshold_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot:   {plot_path}")

    # Summary JSON
    summary = {
        "tau_threshold":        tau,
        "n_seeds":              len(rows),
        "kendalls_tau":         round(float(tau_stat), 6),
        "p_value":              float(p_value),
        "verdict":              verdict,
        "decision_rule":        DECISION_RULE,
        "baseline_mean":        round(float(baseline.mean()),    6),
        "thresholded_mean":     round(float(thresholded.mean()), 6),
        "e_li_under_h0":        round(-1.0 / (37 - 1), 6),  # E[Iᵢ] ≈ −0.028 for n=37
    }
    summary_path = out_dir / "lsap_threshold_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary:{summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    run_name = datetime.now().strftime("lsap_threshold_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"LSAP Threshold Sensitivity — τ = {args.tau}")
    print(f"Seeds: {args.seeds}  Budget: {args.budget}  Output: {out_dir}")
    print()

    rows = run_sensitivity(args)

    if not rows:
        print("ERROR: No results produced.", file=sys.stderr)
        return 1

    # Write CSV
    csv_path = out_dir / "lsap_threshold_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV:    {csv_path}")

    analyze_and_report(rows, args.tau, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
