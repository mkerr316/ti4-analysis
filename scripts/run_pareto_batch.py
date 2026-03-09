#!/usr/bin/env python3
"""
Streaming production batch runner for pareto_optimize().

Memory-safe: after each seed the Pareto front metrics are flushed to CSV and
map tile arrangements serialized to JSON, then the population is explicitly
released and GC collected.  No in-memory accumulation across seeds.

Usage:
    python scripts/run_pareto_batch.py [--seeds N] [--iterations N] \
        [--pop-size N] [--output-dir PATH] [--base-seed N] [--players N]

Outputs (all inside --output-dir / run_YYYYMMDD_HHMMSS/):
    metrics.csv          — one row per (seed, solution) on the Pareto front
    maps/seed_{S}_sol_{K}.json  — tile placement for each Pareto solution
    run_config.json      — CLI params + git hash for reproducibility
"""

import argparse
import csv
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Streaming pareto_optimize() batch runner")
    p.add_argument("--seeds",       type=int, default=1000, help="Number of seeds to run")
    p.add_argument("--iterations",  type=int, default=200,  help="Pareto generations per seed")
    p.add_argument("--pop-size",    type=int, default=12,   help="Population size for pareto_optimize()")
    p.add_argument("--base-seed",   type=int, default=0,    help="First random seed (increments by 1)")
    p.add_argument("--players",     type=int, default=6,    help="Number of players")
    p.add_argument("--output-dir",  type=str, default="output", help="Root output directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Map serialization — tile IDs per hex coordinate
# ---------------------------------------------------------------------------

def serialize_map(ti4_map, score) -> Dict:
    """
    Serialize a TI4Map to a plain dict suitable for JSON.

    Stores the system ID (or None for empty/home spaces) at each hex
    coordinate so the map can be reconstructed from the tile database.
    """
    placements = []
    for space in ti4_map.spaces:
        coord = space.coord
        placements.append({
            "x": int(coord.x),
            "y": int(coord.y),
            "z": int(coord.z),
            "space_type": space.space_type.name,
            "system_id": int(space.system.id) if space.system is not None else None,
        })
    return {
        "placements": placements,
        "score": {
            "balance_gap":  round(float(score.balance_gap),  4),
            "morans_i":     round(float(score.morans_i),     4),
            "jains_index":  round(float(score.jains_index),  4),
            "composite":    round(float(score.composite_score()), 4),
        },
    }


# ---------------------------------------------------------------------------
# CSV row builder
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "seed", "solution_rank",
    "balance_gap", "morans_i", "jains_index", "composite_score",
    "elapsed_sec",
]


def make_row(seed: int, rank: int, score, elapsed: float) -> Dict:
    return {
        "seed":           seed,
        "solution_rank":  rank,
        "balance_gap":    round(float(score.balance_gap),        4),
        "morans_i":       round(float(score.morans_i),           4),
        "jains_index":    round(float(score.jains_index),        4),
        "composite_score":round(float(score.composite_score()),  4),
        "elapsed_sec":    round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Lazy imports — allow CLI --help without heavy deps
    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.spatial_optimizer import pareto_optimize
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

    evaluator = create_joebrew_evaluator()

    # Create run directory
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    maps_dir = run_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Persist config
    config = {
        "run_name":   run_name,
        "seeds":      args.seeds,
        "iterations": args.iterations,
        "pop_size":   args.pop_size,
        "base_seed":  args.base_seed,
        "players":    args.players,
        "started_at": datetime.now().isoformat(),
    }
    try:
        import subprocess
        config["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        config["git_hash"] = "unknown"

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"Seeds: {args.seeds}  Iterations: {args.iterations}  Pop: {args.pop_size}")
    print()

    csv_path = run_dir / "metrics.csv"
    seeds_done = 0
    run_start = time.time()

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        for seed_offset in range(args.seeds):
            seed = args.base_seed + seed_offset
            t0 = time.time()

            try:
                ti4_map = generate_random_map(
                    player_count=args.players,
                    template_name="normal",
                    include_pok=True,
                    random_seed=seed,
                )

                front: List = pareto_optimize(
                    ti4_map,
                    evaluator,
                    iterations=args.iterations,
                    population_size=args.pop_size,
                    random_seed=seed,
                    verbose=False,
                )

                elapsed = time.time() - t0

                # --- Stream metrics to CSV ---
                for rank, (opt_map, score) in enumerate(front):
                    writer.writerow(make_row(seed, rank, score, elapsed))
                csv_file.flush()

                # --- Serialize maps to disk ---
                for rank, (opt_map, score) in enumerate(front):
                    map_path = maps_dir / f"seed_{seed}_sol_{rank}.json"
                    with open(map_path, "w") as mf:
                        json.dump(serialize_map(opt_map, score), mf)

                seeds_done += 1

                # --- ETA ---
                elapsed_total = time.time() - run_start
                avg = elapsed_total / seeds_done
                remaining = (args.seeds - seeds_done) * avg
                front_size = len(front)
                print(
                    f"seed={seed:4d}  front={front_size:2d}  "
                    f"t={elapsed:.1f}s  "
                    f"best_composite={min(s.composite_score() for _,s in front):.2f}  "
                    f"eta={remaining/60:.1f}min"
                )

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"seed={seed:4d}  ERROR after {elapsed:.1f}s: {exc}", file=sys.stderr)
                # Write sentinel row so the seed is traceable
                writer.writerow({
                    "seed": seed, "solution_rank": -1,
                    "balance_gap": float("nan"), "morans_i": float("nan"),
                    "jains_index": float("nan"), "composite_score": float("nan"),
                    "elapsed_sec": round(elapsed, 2),
                })
                csv_file.flush()

            finally:
                # Release population explicitly — critical for 1000-seed runs
                try:
                    del front, ti4_map
                except NameError:
                    pass
                gc.collect()

    total_time = time.time() - run_start
    print()
    print(f"All runs complete. Summary:")
    print(f"  Seeds completed: {seeds_done}/{args.seeds}")
    print(f"  Total time:      {total_time/60:.1f} min")
    print(f"  Metrics CSV:     {csv_path}")
    print(f"  Maps directory:  {maps_dir}")

    return 0 if seeds_done == args.seeds else 1


if __name__ == "__main__":
    sys.exit(main())
