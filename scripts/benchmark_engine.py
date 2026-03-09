#!/usr/bin/env python3
"""
Monte Carlo benchmark: compare Greedy HC, Simulated Annealing, and NSGA-II
on the same random TI4 map seeds.

All three algorithms receive an equal evaluation budget:
    HC:     --hc-iter  evaluations
    SA:     --sa-iter  evaluations  (cooling schedule spans exactly sa-iter steps)
    NSGA-II: --nsga-gen × --nsga-pop evaluations

Memory-safe: maps are deep-copied per algorithm and explicitly released after
each seed.  Results stream to CSV so no data is lost on a crash or interruption.

Usage:
    python scripts/benchmark_engine.py [--seeds N] [--hc-iter N] [--sa-iter N]
        [--nsga-gen N] [--nsga-pop N] [--base-seed N] [--players N]
        [--output-dir PATH] [--algorithms hc,sa,nsga2]

Outputs (inside --output-dir / benchmark_YYYYMMDD_HHMMSS/):
    results.csv      — one row per (seed, algorithm)
    run_config.json  — CLI params + git hash for reproducibility
"""

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo benchmark: HC vs SA vs NSGA-II")
    p.add_argument("--seeds",       type=int, default=100,    help="Number of random seeds")
    p.add_argument("--hc-iter",     type=int, default=1000,   help="HC iterations per seed")
    p.add_argument("--sa-iter",     type=int, default=1000,   help="SA iterations per seed")
    p.add_argument("--nsga-gen",    type=int, default=50,     help="NSGA-II generations")
    p.add_argument("--nsga-pop",    type=int, default=20,     help="NSGA-II population size")
    p.add_argument("--base-seed",   type=int, default=0,      help="First random seed")
    p.add_argument("--players",     type=int, default=6,      help="Number of players")
    p.add_argument("--output-dir",  type=str, default="output", help="Root output directory")
    p.add_argument("--algorithms",  type=str, default="hc,sa,nsga2",
                   help="Comma-separated algorithms to run (hc, sa, nsga2)")
    # SA hyperparameters (from optimize_hyperparameters.py best_params.json)
    p.add_argument("--sa-rate",     type=float, default=0.80,
                   help="SA initial_acceptance_rate (default: 0.80)")
    p.add_argument("--sa-min-temp", type=float, default=0.01,
                   help="SA min_temp (default: 0.01)")
    # NSGA-II hyperparameters
    p.add_argument("--nsga-blob",   type=float, default=0.5,
                   help="NSGA-II blob_fraction (default: 0.5)")
    p.add_argument("--nsga-mut",    type=float, default=0.05,
                   help="NSGA-II mutation_rate (default: 0.05)")
    p.add_argument("--nsga-warm",   type=float, default=0.10,
                   help="NSGA-II warm_fraction (default: 0.10)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "seed", "algorithm",
    "balance_gap", "morans_i", "jains_index", "lisa_penalty",
    "composite_score", "elapsed_sec", "front_size",
]


def make_row(
    seed: int,
    algorithm: str,
    score,
    elapsed: float,
    front_size: int,
) -> Dict:
    return {
        "seed":           seed,
        "algorithm":      algorithm,
        "balance_gap":    round(float(score.balance_gap),        4),
        "morans_i":       round(float(score.morans_i),           4),
        "jains_index":    round(float(score.jains_index),        4),
        "lisa_penalty":   round(float(score.lisa_penalty),       4),
        "composite_score":round(float(score.composite_score()),  4),
        "elapsed_sec":    round(elapsed,                          2),
        "front_size":     front_size,
    }


def make_error_row(seed: int, algorithm: str, elapsed: float) -> Dict:
    return {
        "seed": seed, "algorithm": algorithm,
        "balance_gap": float("nan"), "morans_i": float("nan"),
        "jains_index": float("nan"), "lisa_penalty": float("nan"),
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
    for algo in ("hc", "sa", "nsga2"):
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
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    algos = {a.strip().lower() for a in args.algorithms.split(",")}
    valid = {"hc", "sa", "nsga2"}
    unknown = algos - valid
    if unknown:
        print(f"ERROR: Unknown algorithms: {unknown}. Valid: {valid}", file=sys.stderr)
        return 1

    # Lazy imports — heavy deps only needed at runtime
    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.balance_engine import improve_balance
    from ti4_analysis.algorithms.spatial_optimizer import (
        improve_balance_spatial,
        evaluate_map_multiobjective,
    )
    from ti4_analysis.algorithms.nsga2_optimizer import nsga2_optimize
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

    evaluator = create_joebrew_evaluator()

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
    print(f"HC iterations : {args.hc_iter}")
    print(f"SA iterations : {args.sa_iter}")
    print(f"NSGA-II       : {args.nsga_gen} gen × {args.nsga_pop} pop = "
          f"{args.nsga_gen * args.nsga_pop} evals")
    print()

    # Per-algorithm accumulators for final summary
    accum: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    seeds_done = 0
    run_start = time.time()

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        for seed_offset in range(args.seeds):
            seed = args.base_seed + seed_offset
            seed_start = time.time()

            try:
                initial_map = generate_random_map(
                    player_count=args.players,
                    template_name="normal",
                    include_pok=True,
                    random_seed=seed,
                )

                # ── Greedy Hill Climbing ──────────────────────────────────
                if "hc" in algos:
                    hc_map = initial_map.copy()
                    t0 = time.time()
                    improve_balance(
                        hc_map, evaluator,
                        iterations=args.hc_iter,
                        random_seed=seed,
                    )
                    # improve_balance() returns (gap, history); re-evaluate
                    # for full MultiObjectiveScore via fast path.
                    topo = MapTopology.from_ti4_map(hc_map, evaluator)
                    fs   = FastMapState.from_ti4_map(topo, hc_map, evaluator)
                    hc_score = evaluate_map_multiobjective(hc_map, evaluator, fast_state=fs)
                    elapsed_hc = time.time() - t0

                    row = make_row(seed, "hc", hc_score, elapsed_hc, 1)
                    writer.writerow(row)
                    accum["hc"]["composite_score"].append(row["composite_score"])
                    accum["hc"]["balance_gap"].append(row["balance_gap"])
                    accum["hc"]["elapsed_sec"].append(elapsed_hc)

                    del hc_map, topo, fs

                # ── Simulated Annealing ───────────────────────────────────
                if "sa" in algos:
                    sa_map = initial_map.copy()
                    t0 = time.time()
                    sa_score, _ = improve_balance_spatial(
                        sa_map, evaluator,
                        iterations=args.sa_iter,
                        initial_acceptance_rate=args.sa_rate,
                        min_temp=args.sa_min_temp,
                        random_seed=seed,
                        verbose=False,
                    )
                    elapsed_sa = time.time() - t0

                    row = make_row(seed, "sa", sa_score, elapsed_sa, 1)
                    writer.writerow(row)
                    accum["sa"]["composite_score"].append(row["composite_score"])
                    accum["sa"]["balance_gap"].append(row["balance_gap"])
                    accum["sa"]["elapsed_sec"].append(elapsed_sa)

                    del sa_map

                # ── NSGA-II ───────────────────────────────────────────────
                if "nsga2" in algos:
                    nsga_map = initial_map.copy()
                    t0 = time.time()
                    front = nsga2_optimize(
                        nsga_map, evaluator,
                        generations=args.nsga_gen,
                        population_size=args.nsga_pop,
                        blob_fraction=args.nsga_blob,
                        mutation_rate=args.nsga_mut,
                        warm_fraction=args.nsga_warm,
                        random_seed=seed,
                        verbose=False,
                    )
                    elapsed_nsga = time.time() - t0

                    # Best-composite member for scalar comparison
                    best_score = min(front, key=lambda x: x[1].composite_score())[1]
                    row = make_row(seed, "nsga2", best_score, elapsed_nsga, len(front))
                    writer.writerow(row)
                    accum["nsga2"]["composite_score"].append(row["composite_score"])
                    accum["nsga2"]["balance_gap"].append(row["balance_gap"])
                    accum["nsga2"]["elapsed_sec"].append(elapsed_nsga)

                    del nsga_map, front

                csv_file.flush()
                seeds_done += 1

                elapsed_total = time.time() - run_start
                avg = elapsed_total / seeds_done
                remaining = (args.seeds - seeds_done) * avg
                seed_elapsed = time.time() - seed_start

                print(
                    f"seed={seed:4d}  "
                    + ("" if "hc"    not in algos else f"hc={accum['hc']['composite_score'][-1]:.3f}  ")
                    + ("" if "sa"    not in algos else f"sa={accum['sa']['composite_score'][-1]:.3f}  ")
                    + ("" if "nsga2" not in algos else f"nsga2={accum['nsga2']['composite_score'][-1]:.3f}  ")
                    + f"t={seed_elapsed:.1f}s  eta={remaining/60:.1f}min"
                )

            except Exception as exc:
                elapsed_err = time.time() - seed_start
                print(f"seed={seed:4d}  ERROR after {elapsed_err:.1f}s: {exc}", file=sys.stderr)
                for algo in algos:
                    writer.writerow(make_error_row(seed, algo, elapsed_err))
                csv_file.flush()

            finally:
                try:
                    del initial_map
                except NameError:
                    pass
                gc.collect()

    total_time = time.time() - run_start
    print()
    print(f"Seeds completed : {seeds_done}/{args.seeds}")
    print(f"Total time      : {total_time/60:.1f} min")
    print(f"Results CSV     : {csv_path}")

    print_summary(accum)

    return 0 if seeds_done == args.seeds else 1


if __name__ == "__main__":
    sys.exit(main())
