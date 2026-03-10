#!/usr/bin/env python3
"""
Monte Carlo benchmark: compare Greedy HC, Simulated Annealing, NSGA-II, and
Tabu Search on the same random TI4 map seeds.

All algorithms receive an equal evaluation budget per budget level.
Supports multiprocessing via --workers for embarrassingly parallel speedup
(each seed is independent — no shared state between workers).

Usage:
    python scripts/benchmark_engine.py [--seeds N] [--budgets 1000,5000,10000]
        [--workers 16] [--algorithms hc,sa,nsga2,ts] [--output-dir PATH]

Outputs (inside --output-dir / benchmark_YYYYMMDD_HHMMSS/):
    results.csv      — one row per (seed, algorithm, budget)
    run_config.json  — CLI params + git hash for reproducibility
"""

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo benchmark: HC vs SA vs NSGA-II vs TS")
    p.add_argument("--seeds",       type=int, default=100,    help="Number of random seeds")
    p.add_argument("--hc-iter",     type=int, default=1000,   help="HC iterations per seed")
    p.add_argument("--sa-iter",     type=int, default=1000,   help="SA iterations per seed")
    p.add_argument("--nsga-gen",    type=int, default=50,     help="NSGA-II generations")
    p.add_argument("--nsga-pop",    type=int, default=20,     help="NSGA-II population size")
    p.add_argument("--base-seed",   type=int, default=0,      help="First random seed")
    p.add_argument("--players",     type=int, default=6,      help="Number of players")
    p.add_argument("--output-dir",  type=str, default="output", help="Root output directory")
    p.add_argument("--algorithms",  type=str, default="hc,sa,nsga2",
                   help="Comma-separated algorithms to run (hc, sa, nsga2, ts, rs)")
    p.add_argument("--workers",     type=int, default=1,
                   help="Parallel workers (default: 1 = sequential)")
    p.add_argument("--ts-tenure",   type=int, default=None,
                   help="TS tabu tenure (default: ceil(sqrt(S)))")
    p.add_argument("--sa-rate",     type=float, default=0.80,
                   help="SA initial_acceptance_rate (default: 0.80)")
    p.add_argument("--sa-min-temp", type=float, default=0.01,
                   help="SA min_temp (default: 0.01)")
    p.add_argument("--nsga-blob",   type=float, default=0.5,
                   help="NSGA-II blob_fraction (default: 0.5)")
    p.add_argument("--nsga-mut",    type=float, default=0.05,
                   help="NSGA-II mutation_rate (default: 0.05)")
    p.add_argument("--nsga-warm",   type=float, default=0.10,
                   help="NSGA-II warm_fraction (default: 0.10)")
    p.add_argument("--budgets",     type=str, default=None,
                   help="Comma-separated evaluation budgets for convergence profile "
                        "(e.g. 1000,5000,10000,25000).  When set, overrides "
                        "--hc-iter / --sa-iter / --nsga-gen for each budget level. "
                        "NSGA-II generations = budget // nsga-pop.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "seed", "algorithm", "budget",
    "balance_gap", "morans_i", "jains_index", "jfi_resources", "jfi_influence",
    "lisa_penalty", "composite_score", "elapsed_sec", "front_size",
]


def make_row(
    seed: int,
    algorithm: str,
    score,
    elapsed: float,
    front_size: int,
    budget: int = 0,
) -> Dict:
    return {
        "seed":           seed,
        "algorithm":      algorithm,
        "budget":         budget,
        "balance_gap":    round(float(score.balance_gap),        4),
        "morans_i":       round(float(score.morans_i),           4),
        "jains_index":    round(float(score.jains_index),        4),
        "jfi_resources":  round(float(score.jfi_resources),      4),
        "jfi_influence":  round(float(score.jfi_influence),      4),
        "lisa_penalty":   round(float(score.lisa_penalty),       4),
        "composite_score":round(float(score.composite_score()),  4),
        "elapsed_sec":    round(elapsed,                          2),
        "front_size":     front_size,
    }


def make_error_row(seed: int, algorithm: str, elapsed: float, budget: int = 0) -> Dict:
    return {
        "seed": seed, "algorithm": algorithm, "budget": budget,
        "balance_gap": float("nan"), "morans_i": float("nan"),
        "jains_index": float("nan"), "jfi_resources": float("nan"),
        "jfi_influence": float("nan"), "lisa_penalty": float("nan"),
        "composite_score": float("nan"),
        "elapsed_sec": round(elapsed, 2), "front_size": -1,
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _fmt(values: List[float]) -> str:
    if not values:
        return "     n/a"
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{m:6.3f} ± {s:.3f}"


def print_summary(accum: Dict[str, Dict[str, List[float]]]) -> None:
    header = f"{'Algorithm':<8}  {'composite':>14}  {'balance_gap':>14}  {'elapsed_sec':>14}"
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for algo in ("rs", "hc", "sa", "nsga2", "ts"):
        if algo not in accum:
            continue
        d = accum[algo]
        print(
            f"{algo:<8}  {_fmt(d['composite_score']):>14}  "
            f"{_fmt(d['balance_gap']):>14}  "
            f"{_fmt(d['elapsed_sec']):>14}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Worker (process-safe, self-contained)
# ---------------------------------------------------------------------------

_evaluator = None


def _init_pool():
    """Pool initializer: pin BLAS threads and create the evaluator once per process."""
    global _evaluator
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator
    _evaluator = create_joebrew_evaluator()


def _run_seed(job):
    """Run all algorithms for one (seed, budget) pair.

    Takes a single tuple (for Pool.map compatibility) of primitives only.
    Returns (ok: bool, rows: list[dict]).
    """
    global _evaluator
    (seed, algos_list, budget, hc_iter, sa_iter, nsga_gen,
     players, sa_rate, sa_min_temp, nsga_pop, nsga_blob,
     nsga_mut, nsga_warm, ts_tenure) = job

    algos = set(algos_list)

    import numpy as np

    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.balance_engine import improve_balance
    from ti4_analysis.algorithms.spatial_optimizer import (
        improve_balance_spatial, evaluate_map_multiobjective,
    )
    from ti4_analysis.algorithms.nsga2_optimizer import nsga2_optimize
    from ti4_analysis.algorithms.tabu_search_optimizer import improve_balance_tabu
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState

    evaluator = _evaluator
    rows = []
    seed_start = time.time()

    try:
        initial_map = generate_random_map(
            player_count=players,
            template_name="normal",
            include_pok=True,
            random_seed=seed,
        )

        # ── Random Search (baseline) ──────────────────────────────
        if "rs" in algos:
            rs_map = initial_map.copy()
            t0 = time.time()
            topo = MapTopology.from_ti4_map(rs_map, evaluator)
            fs_base = FastMapState.from_ti4_map(topo, rs_map, evaluator)
            best_rs_score = evaluate_map_multiobjective(rs_map, evaluator, fast_state=fs_base)
            rng_rs = np.random.default_rng(seed)
            S = len(topo.swappable_indices)
            for _ in range(budget - 1):
                fs = fs_base.clone()
                perm = rng_rs.permutation(S)
                for i in range(S - 1, 0, -1):
                    if perm[i] != i:
                        fs.swap(perm[i], i)
                candidate = evaluate_map_multiobjective(rs_map, evaluator, fast_state=fs)
                if candidate.composite_score() < best_rs_score.composite_score():
                    best_rs_score = candidate
                del fs
            rows.append(make_row(seed, "rs", best_rs_score, time.time() - t0, 1, budget))
            del rs_map, topo, fs_base

        if "hc" in algos:
            hc_map = initial_map.copy()
            t0 = time.time()
            improve_balance(
                hc_map, evaluator,
                iterations=hc_iter,
                random_seed=seed,
            )
            topo = MapTopology.from_ti4_map(hc_map, evaluator)
            fs   = FastMapState.from_ti4_map(topo, hc_map, evaluator)
            hc_score = evaluate_map_multiobjective(hc_map, evaluator, fast_state=fs)
            rows.append(make_row(seed, "hc", hc_score, time.time() - t0, 1, budget))
            del hc_map, topo, fs

        if "sa" in algos:
            sa_map = initial_map.copy()
            t0 = time.time()
            sa_score, _ = improve_balance_spatial(
                sa_map, evaluator,
                iterations=sa_iter,
                initial_acceptance_rate=sa_rate,
                min_temp=sa_min_temp,
                random_seed=seed,
                verbose=False,
            )
            rows.append(make_row(seed, "sa", sa_score, time.time() - t0, 1, budget))
            del sa_map

        if "nsga2" in algos:
            nsga_map = initial_map.copy()
            t0 = time.time()
            front = nsga2_optimize(
                nsga_map, evaluator,
                generations=nsga_gen,
                population_size=nsga_pop,
                blob_fraction=nsga_blob,
                mutation_rate=nsga_mut,
                warm_fraction=nsga_warm,
                random_seed=seed,
                verbose=False,
            )
            best_score = min(front, key=lambda x: x[1].composite_score())[1]
            rows.append(make_row(seed, "nsga2", best_score, time.time() - t0, len(front), budget))
            del nsga_map, front

        if "ts" in algos:
            ts_map = initial_map.copy()
            t0 = time.time()
            ts_score, _ = improve_balance_tabu(
                ts_map, evaluator,
                max_evaluations=budget,
                tabu_tenure=ts_tenure,
                random_seed=seed,
                verbose=False,
            )
            rows.append(make_row(seed, "ts", ts_score, time.time() - t0, 1, budget))
            del ts_map

        return True, rows

    except Exception as exc:
        elapsed_err = time.time() - seed_start
        print(f"seed={seed:4d}  ERROR after {elapsed_err:.1f}s: {exc}", file=sys.stderr)
        error_rows = [make_error_row(seed, algo, elapsed_err, budget) for algo in algos]
        return False, error_rows

    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    algos = {a.strip().lower() for a in args.algorithms.split(",")}
    valid = {"hc", "sa", "nsga2", "ts", "rs"}
    unknown = algos - valid
    if unknown:
        print(f"ERROR: Unknown algorithms: {unknown}. Valid: {valid}", file=sys.stderr)
        return 1

    # Initialize evaluator for the main process
    _init_pool()

    # Determine budget schedule
    if args.budgets:
        budget_levels = [int(b.strip()) for b in args.budgets.split(",")]
    else:
        budget_levels = [args.hc_iter]  # single budget, backward-compatible

    # Create run directory
    run_name = datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist config
    config = {
        "run_name":   run_name,
        "seeds":      args.seeds,
        "base_seed":  args.base_seed,
        "players":    args.players,
        "algorithms": sorted(algos),
        "budgets":    budget_levels,
        "workers":    args.workers,
        "hc_iter":    args.hc_iter,
        "sa_iter":    args.sa_iter,
        "nsga_gen":   args.nsga_gen,
        "nsga_pop":   args.nsga_pop,
        "nsga_evals": args.nsga_gen * args.nsga_pop,
        "started_at": datetime.now().isoformat(),
    }
    try:
        import subprocess
        config["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        config["git_hash"] = "unknown"

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    csv_path = run_dir / "results.csv"
    print(f"Run directory : {run_dir}")
    print(f"Seeds         : {args.seeds}  (base_seed={args.base_seed})")
    print(f"Algorithms    : {', '.join(sorted(algos))}")
    print(f"Budget levels : {budget_levels}")
    print(f"Workers       : {args.workers}")
    print()

    seeds_done = 0
    run_start = time.time()
    total_seeds = args.seeds * len(budget_levels)

    pool = None
    if args.workers > 1:
        from multiprocessing import Pool
        pool = Pool(args.workers, initializer=_init_pool)

    try:
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            csv_file.flush()

            for budget in budget_levels:
                if args.budgets:
                    hc_iter  = budget
                    sa_iter  = budget
                    nsga_gen = max(1, budget // args.nsga_pop)
                else:
                    hc_iter  = args.hc_iter
                    sa_iter  = args.sa_iter
                    nsga_gen = args.nsga_gen

                actual_nsga_evals = nsga_gen * args.nsga_pop

                print(f"── Budget {budget:,} ─────────────────────────────────────")
                print(f"   HC={hc_iter}, SA={sa_iter}, "
                      f"NSGA-II={nsga_gen}gen × {args.nsga_pop}pop = {actual_nsga_evals}")

                accum: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

                jobs = [
                    (seed, sorted(algos), budget, hc_iter, sa_iter, nsga_gen,
                     args.players, args.sa_rate, args.sa_min_temp, args.nsga_pop,
                     args.nsga_blob, args.nsga_mut, args.nsga_warm, args.ts_tenure)
                    for seed in range(args.base_seed, args.base_seed + args.seeds)
                ]

                budget_done = 0
                budget_start = time.time()
                mapper = pool.imap_unordered if pool else map

                for ok, rows in mapper(_run_seed, jobs):
                    for row in rows:
                        writer.writerow(row)
                    csv_file.flush()

                    if ok:
                        seeds_done += 1
                        for row in rows:
                            algo = row["algorithm"]
                            accum[algo]["composite_score"].append(row["composite_score"])
                            accum[algo]["balance_gap"].append(row["balance_gap"])
                            accum[algo]["elapsed_sec"].append(row["elapsed_sec"])

                    budget_done += 1
                    elapsed_budget = time.time() - budget_start
                    avg = elapsed_budget / budget_done
                    remaining = (len(jobs) - budget_done) * avg
                    seed_val = rows[0]["seed"] if rows else "?"

                    parts = [f"  seed={seed_val:>4}"]
                    for a in ("rs", "hc", "sa", "nsga2", "ts"):
                        matching = [r for r in rows if r["algorithm"] == a]
                        if matching and not math.isnan(matching[0]["composite_score"]):
                            parts.append(f"{a}={matching[0]['composite_score']:.3f}")
                    parts.append(f"eta={remaining/60:.1f}min")
                    print("  ".join(parts))

                print_summary(accum)

        total_time = time.time() - run_start
        print()
        print(f"Seeds completed : {seeds_done}/{total_seeds}")
        print(f"Total time      : {total_time/60:.1f} min")
        print(f"Results CSV     : {csv_path}")

        return 0 if seeds_done == total_seeds else 1

    finally:
        if pool:
            pool.close()
            pool.join()


if __name__ == "__main__":
    sys.exit(main())
