#!/usr/bin/env python3
"""
Post-hoc LISA validation: verify that minimising the continuous LISA proxy
successfully eliminates *statistically significant* spatial clusters.

For a subset of seeds, re-runs each optimisation algorithm, captures the
final map state, and applies conditional-permutation LISA to count
significant H-H and L-L clusters. Uses 9,999 permutations per location (default)
and reports both per-location (p < alpha) and FDR-corrected (Benjamini–Hochberg, q < 0.05) counts.

Supports multiprocessing via --workers (each seed is independent).

Usage:
    python scripts/validate_lisa_proxy.py --seeds 30 --workers 16
    python scripts/validate_lisa_proxy.py --seeds 30 --algorithms hc,sa,nsga2,ts
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc LISA validation via conditional permutation tests")
    p.add_argument("--seeds", type=int, default=30, help="Number of random seeds (default: 30)")
    p.add_argument("--base-seed", type=int, default=0, help="First random seed")
    p.add_argument("--players", type=int, default=6, help="Number of players")
    p.add_argument("--sa-iter", type=int, default=1000, help="SA iterations per seed")
    p.add_argument("--hc-iter", type=int, default=1000, help="HC iterations per seed")
    p.add_argument("--ts-iter", type=int, default=None,
                   help="TS max_evaluations (default: same as --sa-iter)")
    p.add_argument("--algorithms", type=str, default="hc,sa",
                   help="Comma-separated algorithms (hc, sa, nsga2, ts)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (default: 1 = sequential)")
    p.add_argument("--n-perms", type=int, default=9999,
                   help="Permutations per location for significance test (default: 9999)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Per-location significance threshold (default: 0.05)")
    p.add_argument("--fdr-q", type=float, default=0.05,
                   help="FDR (Benjamini–Hochberg) q level (default: 0.05)")
    p.add_argument("--output-dir", type=str, default="output",
                   help="Root output directory")
    p.add_argument("--sa-rate", type=float, default=0.80)
    p.add_argument("--sa-min-temp", type=float, default=0.01)
    p.add_argument("--nsga-gen", type=int, default=50)
    p.add_argument("--nsga-pop", type=int, default=20)
    p.add_argument("--nsga-blob", type=float, default=0.5)
    p.add_argument("--nsga-mut", type=float, default=0.05)
    p.add_argument("--nsga-warm", type=float, default=0.10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Permutation-tested LISA
# ---------------------------------------------------------------------------

def conditional_permutation_lisa(
    z: np.ndarray,
    W,
    n_perms: int = 9999,
    alpha: float = 0.05,
    fdr_q: float = 0.05,
    rng: np.random.Generator = None,
) -> Dict:
    """
    Conditional permutation test for local Moran's I (Anselin, 1995).

    For each location i, holds z[i] fixed, randomly permutes all other
    values, and recomputes local_I[i] under each permutation to build
    a reference distribution. Reports both per-location significance
    (p < alpha) and FDR-corrected significance (Benjamini–Hochberg, q = fdr_q).

    Returns dict with counts of significant H-H, L-L, H-L, L-H clusters
    (per-location and FDR-corrected).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(z)
    if n < 3:
        return {"n_sig_HH": 0, "n_sig_LL": 0, "n_sig_HL": 0, "n_sig_LH": 0,
                "total_significant": 0, "n_positions": n}

    z_mean = z.mean()
    z_dev = z - z_mean
    m2 = float(z_dev @ z_dev) / n
    if m2 == 0.0:
        return {"n_sig_HH": 0, "n_sig_LL": 0, "n_sig_HL": 0, "n_sig_LH": 0,
                "total_significant": 0, "n_positions": n}

    Wz = np.asarray(W @ z_dev).ravel()
    observed_local_I = z_dev * Wz / m2

    p_values = np.ones(n)
    for i in range(n):
        obs_I = observed_local_I[i]
        count_extreme = 0
        others = np.concatenate([z_dev[:i], z_dev[i+1:]])
        W_row = np.asarray(W[i].toarray()).ravel() if hasattr(W, 'toarray') else W[i]

        for _ in range(n_perms):
            perm = rng.permutation(others)
            perm_full = np.empty(n)
            perm_full[i] = z_dev[i]
            perm_full[:i] = perm[:i]
            perm_full[i+1:] = perm[i:]

            perm_m2 = float(perm_full @ perm_full) / n
            if perm_m2 == 0.0:
                continue
            perm_lag = float(W_row @ perm_full)
            perm_I = perm_full[i] * perm_lag / perm_m2

            if abs(perm_I) >= abs(obs_I):
                count_extreme += 1

        p_values[i] = (count_extreme + 1) / (n_perms + 1)

    significant = p_values < alpha
    n_sig_HH = int(np.sum(significant & (z_dev > 0) & (Wz > 0)))
    n_sig_LL = int(np.sum(significant & (z_dev < 0) & (Wz < 0)))
    n_sig_HL = int(np.sum(significant & (z_dev > 0) & (Wz < 0)))
    n_sig_LH = int(np.sum(significant & (z_dev < 0) & (Wz > 0)))

    # FDR (Benjamini–Hochberg) correction for multiple testing
    from scipy.stats import false_discovery_control
    adjusted = false_discovery_control(p_values, method="bh")
    significant_fdr = adjusted < fdr_q
    n_sig_HH_fdr = int(np.sum(significant_fdr & (z_dev > 0) & (Wz > 0)))
    n_sig_LL_fdr = int(np.sum(significant_fdr & (z_dev < 0) & (Wz < 0)))
    n_sig_HL_fdr = int(np.sum(significant_fdr & (z_dev > 0) & (Wz < 0)))
    n_sig_LH_fdr = int(np.sum(significant_fdr & (z_dev < 0) & (Wz > 0)))

    return {
        "n_sig_HH": n_sig_HH,
        "n_sig_LL": n_sig_LL,
        "n_sig_HL": n_sig_HL,
        "n_sig_LH": n_sig_LH,
        "total_significant": int(significant.sum()),
        "n_sig_HH_fdr": n_sig_HH_fdr,
        "n_sig_LL_fdr": n_sig_LL_fdr,
        "n_sig_HL_fdr": n_sig_HL_fdr,
        "n_sig_LH_fdr": n_sig_LH_fdr,
        "total_significant_fdr": int(significant_fdr.sum()),
        "n_positions": n,
        "lisa_proxy": float(observed_local_I[observed_local_I > 0].sum()),
    }


# ---------------------------------------------------------------------------
# Worker (process-safe, self-contained)
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "seed", "algorithm", "n_sig_HH", "n_sig_LL", "n_sig_HL", "n_sig_LH",
    "total_significant", "n_sig_HH_fdr", "n_sig_LL_fdr", "total_significant_fdr",
    "n_positions", "lisa_proxy", "composite_score", "elapsed_sec",
]

_evaluator = None


def _init_pool():
    """Pool initializer: pin BLAS threads and create the evaluator once per process."""
    global _evaluator
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator
    _evaluator = create_joebrew_evaluator()


def _validate_seed(job):
    """Run all algorithms for one seed and perform permutation-tested LISA.

    Returns list of row dicts (one per algorithm).
    """
    global _evaluator
    (seed, algos_list, sa_iter, hc_iter, ts_iter,
     sa_rate, sa_min_temp,
     nsga_gen, nsga_pop, nsga_blob, nsga_mut, nsga_warm,
     players, n_perms, alpha, fdr_q) = job

    algos = set(algos_list)
    evaluator = _evaluator
    rng = np.random.default_rng(seed)

    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.balance_engine import improve_balance
    from ti4_analysis.algorithms.spatial_optimizer import (
        improve_balance_spatial, evaluate_map_multiobjective,
    )
    from ti4_analysis.algorithms.nsga2_optimizer import nsga2_optimize
    from ti4_analysis.algorithms.tabu_search_optimizer import improve_balance_tabu
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState

    rows = []

    try:
        initial_map = generate_random_map(
            player_count=players, template_name="normal",
            include_pok=True, random_seed=seed,
        )
    except Exception as exc:
        print(f"  seed={seed}: map generation failed: {exc}", file=sys.stderr)
        return rows

    for algo in sorted(algos):
        t0 = time.time()
        try:
            if algo == "hc":
                m = initial_map.copy()
                improve_balance(m, evaluator, iterations=hc_iter, random_seed=seed)
                topo = MapTopology.from_ti4_map(m, evaluator)
                fs = FastMapState.from_ti4_map(topo, m, evaluator)
            elif algo == "sa":
                m = initial_map.copy()
                _, _, _ = improve_balance_spatial(
                    m, evaluator, iterations=sa_iter,
                    initial_acceptance_rate=sa_rate,
                    min_temp=sa_min_temp,
                    random_seed=seed, verbose=False,
                )
                topo = MapTopology.from_ti4_map(m, evaluator)
                fs = FastMapState.from_ti4_map(topo, m, evaluator)
            elif algo == "nsga2":
                m = initial_map.copy()
                front = nsga2_optimize(
                    m, evaluator, generations=nsga_gen,
                    population_size=nsga_pop,
                    blob_fraction=nsga_blob,
                    mutation_rate=nsga_mut,
                    warm_fraction=nsga_warm,
                    random_seed=seed, verbose=False,
                )
                best_map, best_score = min(front, key=lambda x: x[1].composite_score())
                topo = MapTopology.from_ti4_map(best_map, evaluator)
                fs = FastMapState.from_ti4_map(topo, best_map, evaluator)
            elif algo == "ts":
                m = initial_map.copy()
                _, _, _ = improve_balance_tabu(
                    m, evaluator, max_evaluations=ts_iter,
                    random_seed=seed, verbose=False,
                )
                topo = MapTopology.from_ti4_map(m, evaluator)
                fs = FastMapState.from_ti4_map(topo, m, evaluator)
            else:
                continue

            score = evaluate_map_multiobjective(m if algo != "nsga2" else best_map,
                                                evaluator, fast_state=fs)
            z = fs.spatial_values()
            W = topo.spatial_W

            result = conditional_permutation_lisa(z, W, n_perms, alpha, fdr_q, rng)
            elapsed = time.time() - t0

            row = {
                "seed": seed,
                "algorithm": algo,
                "elapsed_sec": round(elapsed, 2),
                "composite_score": round(float(score.composite_score()), 4),
                **{k: result[k] for k in CSV_FIELDS if k in result},
            }
            rows.append(row)

            print(f"  seed={seed:4d}  {algo:<6s}  "
                  f"sig={result['total_significant']}/{result['n_positions']}  "
                  f"FDR_sig={result['total_significant_fdr']}  "
                  f"HH={result['n_sig_HH']} LL={result['n_sig_LL']}  "
                  f"proxy={result['lisa_proxy']:.3f}  "
                  f"t={elapsed:.1f}s")

        except Exception as exc:
            print(f"  seed={seed} {algo}: ERROR {exc}", file=sys.stderr)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    algos = {a.strip().lower() for a in args.algorithms.split(",")}
    ts_iter = args.ts_iter if args.ts_iter is not None else args.sa_iter

    # Initialize evaluator for main process
    _init_pool()

    run_name = datetime.now().strftime("lisa_validation_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "validation_results.csv"

    print(f"LISA Proxy Validation")
    print(f"Seeds       : {args.seeds} (base={args.base_seed})")
    print(f"Algorithms  : {', '.join(sorted(algos))}")
    print(f"Permutations: {args.n_perms}")
    print(f"Workers     : {args.workers}")
    print(f"Output      : {run_dir}")
    print()

    run_start = time.time()

    jobs = [
        (seed, sorted(algos), args.sa_iter, args.hc_iter, ts_iter,
         args.sa_rate, args.sa_min_temp,
         args.nsga_gen, args.nsga_pop, args.nsga_blob, args.nsga_mut, args.nsga_warm,
         args.players, args.n_perms, args.alpha, args.fdr_q)
        for seed in range(args.base_seed, args.base_seed + args.seeds)
    ]

    pool = None
    if args.workers > 1:
        from multiprocessing import Pool
        pool = Pool(args.workers, initializer=_init_pool)

    try:
        all_rows: List[Dict] = []
        mapper = pool.imap_unordered if pool else map

        for rows in mapper(_validate_seed, jobs):
            all_rows.extend(rows)

        # Write CSV (sorted for deterministic output)
        all_rows.sort(key=lambda r: (r["seed"], r["algorithm"]))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

    finally:
        if pool:
            pool.close()
            pool.join()

    total_time = time.time() - run_start
    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Results: {csv_path}")

    try:
        import pandas as pd
        from scipy import stats as scipy_stats
        df = pd.read_csv(csv_path)
        print("\n── Summary (per-location p < alpha) ──")
        summary = df.groupby("algorithm").agg(
            mean_significant=("total_significant", "mean"),
            mean_HH=("n_sig_HH", "mean"),
            mean_LL=("n_sig_LL", "mean"),
            mean_proxy=("lisa_proxy", "mean"),
        ).round(3)
        print(summary)
        if "total_significant_fdr" in df.columns:
            print("\n── Summary (FDR-corrected, q < 0.05) ──")
            summary_fdr = df.groupby("algorithm").agg(
                mean_sig_fdr=("total_significant_fdr", "mean"),
                mean_HH_fdr=("n_sig_HH_fdr", "mean"),
                mean_LL_fdr=("n_sig_LL_fdr", "mean"),
            ).round(3)
            print(summary_fdr)

        # ── Proxy validation: correlation between continuous proxy and cluster count
        proxy_vals = df["lisa_proxy"].values
        sig_counts = df["total_significant"].values
        valid_mask = np.isfinite(proxy_vals) & np.isfinite(sig_counts)
        proxy_valid = proxy_vals[valid_mask]
        sig_valid = sig_counts[valid_mask]

        validation_summary = {}

        if len(proxy_valid) >= 3:
            spearman_r, spearman_p = scipy_stats.spearmanr(proxy_valid, sig_valid)
            pearson_r, pearson_p = scipy_stats.pearsonr(proxy_valid, sig_valid)

            print(f"\n── Proxy Correlation with Significant Cluster Count ──")
            print(f"  Spearman rho = {spearman_r:.4f}  (p = {spearman_p:.2e})")
            print(f"  Pearson  r   = {pearson_r:.4f}  (p = {pearson_p:.2e})")

            validation_summary["spearman_rho"] = round(float(spearman_r), 6)
            validation_summary["spearman_p"] = float(spearman_p)
            validation_summary["pearson_r"] = round(float(pearson_r), 6)
            validation_summary["pearson_p"] = float(pearson_p)

            # Precision analysis: at various proxy thresholds, what fraction have 0 clusters?
            thresholds = [0.5, 1.0, 2.0, 3.0, 5.0]
            print(f"\n── Precision Analysis (proxy < threshold → zero significant clusters?) ──")
            precision_results = []
            for t in thresholds:
                below = proxy_valid < t
                if below.sum() == 0:
                    continue
                n_below = int(below.sum())
                n_zero_sig = int((sig_valid[below] == 0).sum())
                precision = n_zero_sig / n_below
                print(f"  proxy < {t:.1f}: {n_zero_sig}/{n_below} have 0 clusters "
                      f"(precision = {precision:.3f})")
                precision_results.append({
                    "threshold": t,
                    "n_below": n_below,
                    "n_zero_significant": n_zero_sig,
                    "precision": round(precision, 6),
                })
            validation_summary["precision_analysis"] = precision_results

            # Scatter plot: proxy vs significant cluster count
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 6))
                for algo in sorted(df["algorithm"].unique()):
                    mask = df["algorithm"] == algo
                    ax.scatter(
                        df.loc[mask, "lisa_proxy"],
                        df.loc[mask, "total_significant"],
                        alpha=0.6, label=algo.upper(), s=30,
                    )
                ax.set_xlabel("LSAP Continuous Proxy (sum of positive local I)")
                ax.set_ylabel("Significant Clusters (permutation test, p < 0.05)")
                ax.set_title(
                    f"Proxy Validation: Spearman ρ = {spearman_r:.3f}, "
                    f"Pearson r = {pearson_r:.3f}"
                )
                ax.legend()
                ax.grid(True, alpha=0.3)
                scatter_path = run_dir / "proxy_validation_scatter.png"
                fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"\n  Scatter plot → {scatter_path}")
            except Exception as plot_err:
                print(f"  Scatter plot skipped: {plot_err}")

        validation_summary["n_observations"] = int(len(proxy_valid))
        validation_summary["per_algorithm"] = summary.to_dict(orient="index")

        import json
        summary_json_path = run_dir / "proxy_validation_summary.json"
        with open(summary_json_path, "w") as jf:
            json.dump(validation_summary, jf, indent=2, default=str)
        print(f"  Validation JSON → {summary_json_path}")

    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
