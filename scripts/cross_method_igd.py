#!/usr/bin/env python3
"""
Cross-method IGD: project scalar-algorithm terminal states into 3D objective space
and compute IGD+ to the per-budget empirical NSGA-II reference front.

Validates whether single-objective trajectories (SA, TS, HC, SGA) converge near
the Pareto manifold. Outputs cross_method_igd.csv and an optional report.

Usage:
    python scripts/cross_method_igd.py --run-dir output/benchmark_YYYYMMDD_HHMMSS
    python scripts/cross_method_igd.py --results-csv path/results.csv --archive-dir path/pareto_archives --output-dir path
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def igd_plus(front: np.ndarray, reference_front: np.ndarray) -> float:
    """
    IGD+ (Ishibuchi et al. 2015): Pareto-compliant.
    For each reference point, minimum modified distance to any point in the front.
    """
    if len(front) == 0 or len(reference_front) == 0:
        return float("inf")
    total = 0.0
    for r in reference_front:
        min_dist = float("inf")
        for p in front:
            diff = np.maximum(p - r, 0.0)
            d = float(np.sqrt(np.sum(diff**2)))
            if d < min_dist:
                min_dist = d
        total += min_dist
    return total / len(reference_front)


def load_pareto_archive(csv_path: Path) -> np.ndarray:
    """Load archive into (N, 3) minimisation objectives: (1-JFI, |morans_i|, lisa_penalty)."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            jfi = float(row["jains_index"])
            mi = float(row["morans_i"])
            lp = float(row["lisa_penalty"])
            rows.append([1.0 - jfi, abs(mi), lp])
    return np.array(rows) if rows else np.empty((0, 3))


def build_ref_by_budget(archive_dir: Path) -> dict:
    """
    Discover pareto_archive_seed*_b*.csv; parse budget; stack points per budget.
    Returns dict: budget -> (M, 3) array of reference points (minimisation space).
    """
    pattern = re.compile(r"pareto_archive_seed\d+_b(\d+)\.csv$")
    ref_by_budget = defaultdict(list)
    for f in archive_dir.glob("pareto_archive_seed*.csv"):
        m = pattern.search(f.name)
        if m:
            budget = int(m.group(1))
            pts = load_pareto_archive(f)
            if len(pts) > 0:
                ref_by_budget[budget].append(pts)
    out = {}
    for budget, lists in ref_by_budget.items():
        out[budget] = np.vstack(lists)
    return out


def row_to_point(row: dict) -> np.ndarray:
    """Map results.csv row to 3D minimisation objective vector."""
    jfi = float(row["jains_index"])
    mi = float(row["morans_i"])
    lp = float(row["lisa_penalty"])
    return np.array([1.0 - jfi, abs(mi), lp], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-method IGD: scalar terminal states vs NSGA-II reference front"
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory containing results.csv and pareto_archives/",
    )
    p.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Path to results.csv (use with --archive-dir and --output-dir)",
    )
    p.add_argument(
        "--archive-dir",
        type=str,
        default=None,
        help="Directory containing pareto_archive_seed*_b*.csv",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for cross_method_igd.csv (default: run-dir or archive-dir)",
    )
    p.add_argument(
        "--algorithms",
        type=str,
        default="sa,ts,hc,sga",
        help="Comma-separated algorithm IDs to include (default: sa,ts,hc,sga)",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Print mean (and std) IGD+ per (algorithm, budget) to stdout",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    algos = {a.strip().lower() for a in args.algorithms.split(",")}

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"ERROR: run-dir does not exist: {run_dir}", file=sys.stderr)
            return 1
        results_csv = run_dir / "results.csv"
        archive_dir = run_dir / "pareto_archives"
        output_dir = run_dir
    else:
        if not args.results_csv or not args.archive_dir:
            print(
                "ERROR: provide --run-dir or both --results-csv and --archive-dir",
                file=sys.stderr,
            )
            return 1
        results_csv = Path(args.results_csv)
        archive_dir = Path(args.archive_dir)
        output_dir = Path(args.output_dir) if args.output_dir else archive_dir

    if not results_csv.exists():
        print(f"ERROR: results CSV not found: {results_csv}", file=sys.stderr)
        return 1
    if not archive_dir.exists():
        print(f"ERROR: archive dir not found: {archive_dir}", file=sys.stderr)
        return 1

    ref_by_budget = build_ref_by_budget(archive_dir)
    if not ref_by_budget:
        print(
            "ERROR: no pareto_archive_seed*_b*.csv files found in archive dir",
            file=sys.stderr,
        )
        return 1

    rows_out = []
    report_vals = defaultdict(list)  # (algo, budget) -> [igd+ values]

    with open(results_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            algo = row.get("algorithm", "").strip().lower()
            if algo not in algos:
                continue
            try:
                budget = int(row["budget"])
            except (KeyError, ValueError):
                continue
            if budget not in ref_by_budget:
                continue
            point = row_to_point(row)
            ref = ref_by_budget[budget]
            igdp = igd_plus(np.array([point]), ref)
            rows_out.append({
                "seed": row.get("seed", ""),
                "budget": budget,
                "chain_id": row.get("chain_id", 0),
                "algorithm": algo,
                "igd_plus": round(igdp, 6),
            })
            report_vals[(algo, budget)].append(igdp)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cross_method_igd.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["seed", "budget", "chain_id", "algorithm", "igd_plus"],
        )
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {out_path}")

    if args.report and report_vals:
        import statistics
        print("\n--- Cross-method IGD+ report (mean, std) ---")
        for (algo, budget) in sorted(report_vals.keys()):
            vals = report_vals[(algo, budget)]
            mean_igd = statistics.mean(vals)
            std_igd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {algo} @ budget={budget}: mean={mean_igd:.6f}  std={std_igd:.6f}  n={len(vals)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
