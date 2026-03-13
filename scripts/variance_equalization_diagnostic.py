#!/usr/bin/env python3
"""
Variance equalization diagnostic for the 5:5:3 objective weights.

Addresses the reviewer concern that if the empirical variance of LSAP across the
map space is substantially larger than the variance of JFI or Moran's I, LSAP will
silently dominate the Markov chain acceptance dynamics regardless of the nominal
5:5:3 weights.

This script computes the standard deviation of the three normalized objective terms:
  - hinge_morans_i  = max(0, I − E[I]) = max(0, I + 1/(n-1))
  - jfi_gap         = 1 − J_min
  - lisa_norm       = LSAP / (n × (n-1))

at two points:
  1. Generation 0: across a sample of N_RANDOM unoptimized maps (random configurations).
  2. Convergence: across a sample of SA-optimized solutions.

Reports the empirical σ at each stage, the ratio to the nominal 5:5:3 weights, and
a verdict on whether any term dominates the Markov chain dynamics.

Usage:
    python scripts/variance_equalization_diagnostic.py \\
        [--n-random 1000] [--n-sa 50] [--sa-budget 2000] [--players 6] \\
        [--base-seed 0] [--output-dir output/]
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Nominal weights (5:5:3 ratios, sum to 1.0)
# ---------------------------------------------------------------------------

NOMINAL_WEIGHTS = {
    "hinge_morans_i": 5.0 / 13.0,
    "jfi_gap":        5.0 / 13.0,
    "lisa_norm":      3.0 / 13.0,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Variance equalization diagnostic for 5:5:3 objective weights"
    )
    p.add_argument("--n-random",   type=int,   default=1000,
                   help="Number of unoptimized maps to sample (Gen-0 proxy; default 1000)")
    p.add_argument("--n-sa",       type=int,   default=50,
                   help="Number of SA runs for convergence sample (default 50)")
    p.add_argument("--sa-budget",  type=int,   default=2000,
                   help="SA evaluation budget per run (default 2000)")
    p.add_argument("--players",    type=int,   default=6,
                   help="Number of players (default 6)")
    p.add_argument("--base-seed",  type=int,   default=0,
                   help="First random seed (default 0)")
    p.add_argument("--output-dir", type=str,   default="output",
                   help="Root output directory (default output/)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_terms(score) -> dict:
    """Extract the three normalized objective terms from a MultiObjectiveScore."""
    n = score.n_spatial
    hinge = max(0.0, score.morans_i + 1.0 / max(1, n - 1))
    jfi_gap = 1.0 - score.jains_index
    lisa_norm = score.lisa_penalty / max(1, n * (n - 1))
    return {
        "hinge_morans_i": hinge,
        "jfi_gap":        jfi_gap,
        "lisa_norm":      lisa_norm,
    }


def _sigma_table(records: list[dict]) -> dict:
    """Compute mean, std, and weighted std for each term across a list of records."""
    result = {}
    for key in ("hinge_morans_i", "jfi_gap", "lisa_norm"):
        vals = np.array([r[key] for r in records])
        result[key] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    return result


def _dominance_verdict(sigma: dict) -> str:
    """
    Compare weighted empirical stds.  If one term's weighted σ exceeds the
    others by a factor of 2 or more, it effectively dominates acceptance dynamics.
    """
    weighted = {k: sigma[k]["std"] * NOMINAL_WEIGHTS[k] for k in sigma}
    total = sum(weighted.values()) or 1.0
    shares = {k: weighted[k] / total for k in weighted}

    dominant = max(shares, key=shares.get)
    dominant_share = shares[dominant]

    if dominant_share > 0.60:
        return (
            f"WARNING — '{dominant}' accounts for {dominant_share:.1%} of weighted variance. "
            "This term may dominate Markov chain acceptance dynamics; "
            "the nominal 5:5:3 weights may not reflect operational priorities."
        )
    else:
        return (
            "OK — no single term dominates (largest share: "
            f"'{dominant}' at {dominant_share:.1%}). "
            "The 5:5:3 nominal weights appear operationally consistent."
        )


def _print_table(label: str, sigma: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  {'Term':<20s}  {'Mean':>8s}  {'Std':>8s}  {'Nom.W':>7s}  {'w×σ':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}")
    for key in ("hinge_morans_i", "jfi_gap", "lisa_norm"):
        w = NOMINAL_WEIGHTS[key]
        s = sigma[key]["std"]
        print(f"  {key:<20s}  {sigma[key]['mean']:>8.5f}  {s:>8.5f}  "
              f"{w:>7.4f}  {w*s:>8.5f}")
    print(f"\n  Verdict: {_dominance_verdict(sigma)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState
    from ti4_analysis.algorithms.spatial_optimizer import (
        evaluate_map_multiobjective,
        improve_balance_spatial,
    )
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

    evaluator = create_joebrew_evaluator()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Variance Equalization Diagnostic — {timestamp}")
    print(f"  Gen-0 sample size : {args.n_random} random maps")
    print(f"  Convergence sample: {args.n_sa} SA runs × {args.sa_budget} evals")
    print(f"  Players           : {args.players}")
    print(f"  Base seed         : {args.base_seed}")

    # ── Generation-0 sample (random unoptimized configurations) ──────────────
    print(f"\nSampling {args.n_random} random configurations (Gen-0)...")
    gen0_records: list[dict] = []
    for i in range(args.n_random):
        seed = args.base_seed + i
        try:
            ti4_map = generate_random_map(player_count=args.players, random_seed=seed)
            topology = MapTopology.from_ti4_map(ti4_map, evaluator)
            fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)
            score = evaluate_map_multiobjective(ti4_map, evaluator, fast_state=fast_state)
            rec = _extract_terms(score)
            rec["seed"] = seed
            gen0_records.append(rec)
        except Exception as exc:
            print(f"  [seed={seed}] ERROR: {exc}", file=sys.stderr)
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{args.n_random} done")

    gen0_sigma = _sigma_table(gen0_records)
    _print_table(f"GEN-0 (random configurations, n={len(gen0_records)})", gen0_sigma)

    # ── Convergence sample (SA-optimized solutions) ────────────────────────
    print(f"\nRunning {args.n_sa} SA optimizations (convergence sample)...")
    conv_records: list[dict] = []
    for i in range(args.n_sa):
        seed = args.base_seed + 10_000 + i   # distinct from Gen-0 seeds
        try:
            ti4_map = generate_random_map(player_count=args.players, random_seed=seed)
            t0 = time.time()
            best_map, best_score = improve_balance_spatial(
                ti4_map,
                evaluator,
                max_iterations=args.sa_budget,
            )
            elapsed = time.time() - t0
            rec = _extract_terms(best_score)
            rec["seed"] = seed
            rec["elapsed_sec"] = round(elapsed, 2)
            conv_records.append(rec)
        except Exception as exc:
            print(f"  [seed={seed}] ERROR: {exc}", file=sys.stderr)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{args.n_sa} done")

    conv_sigma = _sigma_table(conv_records)
    _print_table(f"CONVERGENCE (SA-optimized solutions, n={len(conv_records)})", conv_sigma)

    # ── σ shift summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  σ SHIFT (Gen-0 → Convergence)")
    print(f"{'=' * 60}")
    print(f"  {'Term':<20s}  {'σ Gen-0':>10s}  {'σ Conv.':>10s}  {'Ratio':>8s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")
    for key in ("hinge_morans_i", "jfi_gap", "lisa_norm"):
        s0 = gen0_sigma[key]["std"]
        sc = conv_sigma[key]["std"]
        ratio = sc / s0 if s0 > 0 else float("nan")
        flag = " ← shrinks" if ratio < 0.5 else (" ← inflates" if ratio > 2.0 else "")
        print(f"  {key:<20s}  {s0:>10.5f}  {sc:>10.5f}  {ratio:>8.3f}{flag}")

    # ── Write CSV ──────────────────────────────────────────────────────────
    csv_path = output_dir / f"variance_equalization_{timestamp}.csv"
    fieldnames = ["phase", "seed", "hinge_morans_i", "jfi_gap", "lisa_norm"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rec in gen0_records:
            w.writerow({**rec, "phase": "gen0"})
        for rec in conv_records:
            w.writerow({**rec, "phase": "convergence"})
    print(f"\nRaw data written to: {csv_path}")


if __name__ == "__main__":
    main()
