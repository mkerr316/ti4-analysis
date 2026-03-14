#!/usr/bin/env python3
"""
Unified Hypervolume (HV) analysis for all algorithms.

Uses the empirical Pareto fronts saved by `benchmark_engine.py` in
`unified_archives/` to compute HV for each (algorithm, seed, budget) in the
common 3-objective space:

    (1 − jains_index, |morans_i|, lisa_penalty)

This places scalar algorithms (RS, HC, SA, SGA, TS) and NSGA-II on the same
objective footing. For each run we:

  1. Load its empirical front (non-dominated MultiObjectiveScore snapshots
     extracted from the logged run history).
  2. Transform to minimisation objectives as above.
  3. Compute Hypervolume against a common reference point (auto = worst
     observed across all algorithms × 1.1, or user-specified).

Outputs:
  - unified_hv.csv          — one row per (algorithm, seed, budget)
  - unified_hv_report.txt   — summary + Friedman/Wilcoxon/Vargha–Delaney on HV
                               at a selected budget (optional)

Usage examples:
    # Basic HV table for all budgets
    python scripts/unified_hv_analysis.py --archive-dir output/benchmark_*/unified_archives

    # HV + non-parametric stats at budget 500000
    python scripts/unified_hv_analysis.py --archive-dir output/benchmark_*/unified_archives \\
        --budget 500000
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ── Hypervolume (HV) — copied in minimal form to avoid circular imports ─────

def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    dominated = np.all(points <= ref, axis=1)
    pts = points[dominated]
    if len(pts) == 0:
        return 0.0
    sorted_pts = pts[pts[:, 0].argsort()]
    hv = 0.0
    prev_y = ref[1]
    for p in sorted_pts:
        hv += (ref[0] - p[0]) * (prev_y - p[1])
        prev_y = min(prev_y, p[1])
    return float(hv)


def _hv2d_exact(pts2d: np.ndarray, ref2: np.ndarray) -> float:
    """Exact 2D hypervolume via sweep-line with non-dominated filtering."""
    if len(pts2d) == 0:
        return 0.0
    nondom = np.ones(len(pts2d), dtype=bool)
    for i in range(len(pts2d)):
        for j in range(len(pts2d)):
            if i != j and np.all(pts2d[j] <= pts2d[i]) and np.any(pts2d[j] < pts2d[i]):
                nondom[i] = False
                break
    pts2d = pts2d[nondom]
    if len(pts2d) == 0:
        return 0.0
    pts2d = pts2d[pts2d[:, 0].argsort()]
    hv = 0.0
    prev_y = ref2[1]
    for i in range(len(pts2d) - 1, -1, -1):
        if pts2d[i][1] < prev_y:
            hv += (ref2[0] - pts2d[i][0]) * (prev_y - pts2d[i][1])
            prev_y = pts2d[i][1]
    return hv


def _hv3d_exact(points: np.ndarray, ref: np.ndarray) -> float:
    """
    Exact 3D Hypervolume via sweep-line (WFG/HSO-style).

    points: (N, 3) array already in minimisation-objective space
            (1-JFI, |Moran's I|, lisa_penalty) — as produced by
            load_unified_archive().
    ref:    (3,) reference point (all objectives must be dominated by ref).

    Replaces the previous Monte Carlo approximation (hypervolume_mc).
    """
    if len(points) == 0:
        return 0.0
    pts = points[np.all(points < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[pts[:, 2].argsort()]
    hv = 0.0
    prev_z = ref[2]
    for i in range(len(pts) - 1, -1, -1):
        slice_depth = prev_z - pts[i][2]
        if slice_depth <= 0:
            prev_z = pts[i][2]
            continue
        hv += _hv2d_exact(pts[: i + 1, :2], ref[:2]) * slice_depth
        prev_z = pts[i][2]
    return hv


def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    if points.shape[1] == 2:
        return hypervolume_2d(points, ref)
    return _hv3d_exact(points, ref)


# ── Vargha–Delaney A (stochastic superiority) ───────────────────────────────

def vargha_delaney_a(x: np.ndarray, y: np.ndarray) -> float:
    """
    Vargha–Delaney A statistic: A = P(X < Y) + 0.5·P(X == Y).

    Here we treat *higher HV as better*. When interpreting A:
        A ≈ 0.50 → no difference
        A > 0.56 → small effect   (X tends to have higher HV)
        A > 0.64 → medium effect
        A > 0.71 → large effect
    """
    n = len(x)
    m = len(y)
    # Expand to pairwise comparisons
    count_greater = 0
    count_equal = 0
    for xi in x:
        count_greater += np.sum(xi > y)
        count_equal += np.sum(xi == y)
    return float(count_greater + 0.5 * count_equal) / (n * m)


def vda_magnitude(a: float) -> str:
    d = abs(a - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified HV analysis for all algorithms")
    p.add_argument(
        "--archive-dir",
        type=str,
        required=True,
        help="Directory containing unified_archive_algo*_seed*_b*.csv files "
             "(produced by benchmark_engine.py)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as archive-dir)",
    )
    p.add_argument(
        "--reference",
        type=str,
        default="auto",
        help="Reference point: 'auto' (worst observed + 10%%) or comma-separated values",
    )
    p.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="[DEPRECATED] No-op. Exact 3D sweep-line is used; Monte Carlo estimation "
             "has been removed.",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Optional: single budget level for statistical tests (e.g. 500000). "
             "If omitted, Friedman/Wilcoxon are not run.",
    )
    return p.parse_args()


def load_unified_archive(csv_path: Path) -> np.ndarray:
    """
    Load a unified archive CSV into an (N, 3) array of objective values.

    Expected columns: jains_index, morans_i, lisa_penalty
    Converts to minimisation objectives: (1-JFI, |morans_i|, lisa_penalty).
    """
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            jfi = float(row["jains_index"])
            mi = float(row["morans_i"])
            lp = float(row["lisa_penalty"])
            rows.append([1.0 - jfi, abs(mi), lp])
    return np.array(rows) if rows else np.empty((0, 3))


def parse_meta_from_stem(stem: str) -> Tuple[str, int, int]:
    """
    Parse algorithm, seed, budget from unified archive filename stem:
        unified_archive_algo{algo}_seed{seed}_b{budget}
    """
    if not stem.startswith("unified_archive_algo"):
        raise ValueError(f"Unexpected archive stem: {stem}")
    rest = stem[len("unified_archive_algo"):]  # e.g. "sa_seed0_b1000"
    parts = rest.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected archive stem format: {stem}")
    algo = parts[0]
    seed_str = parts[1]
    budget_str = parts[2]
    if not seed_str.startswith("seed") or not budget_str.startswith("b"):
        raise ValueError(f"Unexpected archive stem format: {stem}")
    seed = int(seed_str.replace("seed", ""))
    budget = int(budget_str.replace("b", ""))
    return algo, seed, budget


def run_stats(hv_table: List[Dict], budget: int, out_path: Path) -> None:
    """
    Run Friedman + pairwise Wilcoxon + Vargha–Delaney on HV at a single budget.
    """
    # Filter to chosen budget
    rows = [r for r in hv_table if r["budget"] == budget]
    if not rows:
        print(f"No HV rows found at budget={budget}", file=sys.stderr)
        return

    # Group by algorithm
    algos = sorted({r["algorithm"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    # Build wide matrix: seeds × algos
    data: Dict[str, Dict[int, float]] = defaultdict(dict)
    for r in rows:
        data[r["algorithm"]][r["seed"]] = r["hv"]

    # Ensure complete data
    for a in algos:
        if len(data[a]) != len(seeds):
            print(f"WARNING: algorithm {a} has {len(data[a])} seeds at budget {budget}, "
                  f"expected {len(seeds)}", file=sys.stderr)

    # Friedman test (repeated measures)
    samples = [np.array([data[a][s] for s in seeds if s in data[a]]) for a in algos]
    friedman_stat, friedman_p = stats.friedmanchisquare(*samples)

    lines: List[str] = []
    w = lines.append
    rule = "=" * 90

    w(rule)
    w(f"UNIFIED HV ANALYSIS — BUDGET {budget}")
    w(rule)
    w("")

    # Descriptive stats
    w("1. DESCRIPTIVE STATISTICS (Hypervolume, higher = better)")
    w("-" * 90)
    for a in algos:
        vals = np.array([data[a][s] for s in seeds if s in data[a]])
        w(f"  {a:<8s}  median={np.median(vals):.6f}  "
          f"IQR=[{np.quantile(vals, 0.25):.6f}, {np.quantile(vals, 0.75):.6f}]  "
          f"min={vals.min():.6f}  max={vals.max():.6f}  n={len(vals)}")

    # Friedman
    w(f"\n{rule}")
    w("2. FRIEDMAN OMNIBUS TEST (HV, higher = better)")
    w("-" * 90)
    w(f"  chi2 = {friedman_stat:.4f}, df = {len(algos) - 1}, p = {friedman_p:.6g}")

    # Pairwise Wilcoxon + Vargha–Delaney
    w(f"\n{rule}")
    w("3. PAIRWISE WILCOXON SIGNED-RANK + VARGHA–DELANEY A (HV)")
    w("-" * 90)
    pairs = []
    raw_pvalues = []
    stats_rows = []
    for i, a in enumerate(algos):
        for b in algos[i + 1:]:
            x = np.array([data[a][s] for s in seeds if s in data[a] and s in data[b]])
            y = np.array([data[b][s] for s in seeds if s in data[a] and s in data[b]])
            if len(x) == 0 or len(y) == 0:
                continue
            # Signed-rank on differences (paired seeds)
            try:
                W_stat, p_val = stats.wilcoxon(x, y, alternative="two-sided")
            except ValueError:
                W_stat, p_val = np.nan, 1.0
            a_stat = vargha_delaney_a(x, y)
            pairs.append((a, b))
            raw_pvalues.append(p_val)
            stats_rows.append((a, b, W_stat, p_val, a_stat))

    if raw_pvalues:
        reject, corrected, _, _ = multipletests(raw_pvalues, method="holm")
        for (a, b), (W_stat, p_raw, a_stat), p_corr, sig in zip(
            pairs, stats_rows, corrected, reject
        ):
            mag = vda_magnitude(a_stat)
            w(f"  {a:<4s} vs {b:<4s}  "
              f"W={W_stat:.4f}  p_raw={p_raw:.4g}  p_holm={p_corr:.4g}  "
              f"A={a_stat:.3f} ({mag}), significant={bool(sig)}")

    out_path.write_text("\n".join(lines))
    print(f"\nUnified HV report written to {out_path}")


def main() -> int:
    args = parse_args()
    archive_dir = Path(args.archive_dir)
    if not archive_dir.exists():
        print(f"ERROR: {archive_dir} does not exist", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else archive_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hv_files = sorted(archive_dir.glob("unified_archive_algo*_seed*_b*.csv"))
    if not hv_files:
        print(f"No unified_archive_algo*_seed*_b*.csv files found in {archive_dir}",
              file=sys.stderr)
        return 1

    print(f"Loading {len(hv_files)} unified archives from {archive_dir}")

    all_points = []
    fronts: List[Tuple[str, int, int, np.ndarray]] = []

    for f in hv_files:
        algo, seed, budget = parse_meta_from_stem(f.stem)
        front = load_unified_archive(f)
        if front.size == 0:
            continue
        fronts.append((algo, seed, budget, front))
        all_points.append(front)

    if not all_points:
        print("No valid unified fronts found", file=sys.stderr)
        return 1

    combined = np.vstack(all_points)

    if args.reference == "auto":
        worst = combined.max(axis=0)
        ref = worst * 1.1
    else:
        ref = np.array([float(x) for x in args.reference.split(",")])

    print(f"Reference point: {ref}")
    print(f"Combined reference front: {len(combined)} points")

    if args.mc_samples is not None:
        print("WARNING: --mc-samples is deprecated and has no effect. "
              "Exact 3D sweep-line HV is used.", file=sys.stderr)

    hv_table: List[Dict] = []
    for algo, seed, budget, front in fronts:
        hv = hypervolume(front, ref)
        hv_table.append({
            "algorithm": algo,
            "seed": seed,
            "budget": budget,
            "hv": float(hv),
        })

    # Write CSV
    csv_path = output_dir / "unified_hv.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "seed", "budget", "hv"])
        writer.writeheader()
        for row in sorted(hv_table, key=lambda r: (r["budget"], r["algorithm"], r["seed"])):
            writer.writerow(row)
    print(f"Unified HV CSV written to {csv_path}")

    # Optional stats at a chosen budget
    if args.budget is not None:
        report_path = output_dir / f"unified_hv_stats_budget{args.budget}.txt"
        run_stats(hv_table, args.budget, report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

