#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for the TI4 map optimization algorithms.

Tunes the free parameters of either Simulated Annealing or NSGA-II using
Bayesian optimization (TPE sampler).  Evaluation seeds are deliberately
disjoint from benchmark seeds (default base_seed=9000 vs benchmark 0-999)
to prevent overfitting the hyperparameters to specific map layouts.

Usage:
    python scripts/optimize_hyperparameters.py --algo sa   [options]
    python scripts/optimize_hyperparameters.py --algo nsga2 [options]

SA search space:
    initial_acceptance_rate  [0.50, 0.99]   — controls T₀ magnitude
    min_temp                 [0.001, 0.10]  — controls exploitation depth

NSGA-II search space:
    blob_fraction            [0.25, 0.75]   — BFS crossover blob size
    mutation_rate            [0.01, 0.30]   — per-position swap probability
    warm_fraction            [0.00, 0.30]   — Gen-0 warm-start fraction

Outputs (inside --output-dir / optuna_YYYYMMDD_HHMMSS/):
    best_params.json  — optimal hyperparameters + best objective value
    trials.csv        — all trial results (for convergence plots)
    run_config.json   — CLI params + git hash for reproducibility
"""

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter tuning for TI4 optimizers")
    p.add_argument("--algo",        required=True, choices=["sa", "nsga2"],
                   help="Algorithm to tune")
    p.add_argument("--trials",      type=int, default=50,
                   help="Number of Optuna trials (default: 50)")
    p.add_argument("--eval-seeds",  type=int, default=15,
                   help="Seeds per trial for mean estimation (default: 15)")
    p.add_argument("--iter-budget", type=int, default=200,
                   help="Iteration budget per seed: SA=sa_iter, NSGA2=gen×pop (default: 200)")
    p.add_argument("--base-seed",   type=int, default=9000,
                   help="Base random seed for tuning — keep disjoint from benchmark seeds (default: 9000)")
    p.add_argument("--players",     type=int, default=6,
                   help="Number of players (default: 6)")
    p.add_argument("--output-dir",  type=str, default="output",
                   help="Root output directory (default: output)")
    p.add_argument("--study-name",  type=str, default=None,
                   help="Optuna study name (enables resume via --storage)")
    p.add_argument("--storage",     type=str, default=None,
                   help="Optuna storage URL, e.g. sqlite:///optuna.db")
    return p.parse_args()


# ---------------------------------------------------------------------------
# NSGA-II budget factorization
# ---------------------------------------------------------------------------

def _nsga2_budget(iter_budget: int):
    """
    Factorize iter_budget into (generations, pop_size) with pop ≥ 10, gen ≥ 5.

    Example: iter_budget=200 → pop=14, gen=14 (196 evals).
    """
    pop_size    = max(10, int(math.sqrt(iter_budget)))
    generations = max(5,  iter_budget // pop_size)
    return generations, pop_size


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def objective_sa(trial, args, evaluator, generate_random_map, improve_balance_spatial):
    """Objective for SA: mean composite_score over eval_seeds random maps."""
    rate  = trial.suggest_float("initial_acceptance_rate", 0.50, 0.99)
    min_t = trial.suggest_float("min_temp", 0.001, 0.10, log=True)

    scores = []
    for i in range(args.eval_seeds):
        seed = args.base_seed + i
        m = generate_random_map(
            player_count=args.players,
            template_name="normal",
            include_pok=True,
            random_seed=seed,
        )
        score, _ = improve_balance_spatial(
            m, evaluator,
            iterations=args.iter_budget,
            initial_acceptance_rate=rate,
            min_temp=min_t,
            random_seed=seed,
            verbose=False,
        )
        scores.append(score.composite_score())
        del m
        gc.collect()

    return statistics.mean(scores)


def objective_nsga2(trial, args, evaluator, generate_random_map, nsga2_optimize):
    """Objective for NSGA-II: mean best-composite over eval_seeds random maps."""
    blob_frac  = trial.suggest_float("blob_fraction",  0.25, 0.75)
    mut_rate   = trial.suggest_float("mutation_rate",  0.01, 0.30, log=True)
    warm_frac  = trial.suggest_float("warm_fraction",  0.00, 0.30)

    generations, pop_size = _nsga2_budget(args.iter_budget)

    scores = []
    for i in range(args.eval_seeds):
        seed = args.base_seed + i
        m = generate_random_map(
            player_count=args.players,
            template_name="normal",
            include_pok=True,
            random_seed=seed,
        )
        front = nsga2_optimize(
            m, evaluator,
            generations=generations,
            population_size=pop_size,
            blob_fraction=blob_frac,
            mutation_rate=mut_rate,
            warm_fraction=warm_frac,
            random_seed=seed,
            verbose=False,
        )
        best = min(front, key=lambda x: x[1].composite_score())[1]
        scores.append(best.composite_score())
        del m, front
        gc.collect()

    return statistics.mean(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Check Optuna availability before doing anything heavy
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print(
            "ERROR: Optuna is not installed.\n"
            "Install it with:  pip install optuna\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        return 1

    # Lazy imports of heavy TI4 deps
    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.spatial_optimizer import improve_balance_spatial
    from ti4_analysis.algorithms.nsga2_optimizer import nsga2_optimize
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

    evaluator = create_joebrew_evaluator()

    # Create run directory
    run_name = datetime.now().strftime("optuna_%Y%m%d_%H%M%S")
    run_dir  = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist config
    gen, pop = _nsga2_budget(args.iter_budget)
    config = {
        "run_name":   run_name,
        "algo":       args.algo,
        "trials":     args.trials,
        "eval_seeds": args.eval_seeds,
        "iter_budget": args.iter_budget,
        "nsga2_gen":  gen,
        "nsga2_pop":  pop,
        "base_seed":  args.base_seed,
        "players":    args.players,
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

    csv_path = run_dir / "trials.csv"

    print(f"Run directory : {run_dir}")
    print(f"Algorithm     : {args.algo.upper()}")
    print(f"Trials        : {args.trials}")
    print(f"Eval seeds    : {args.eval_seeds}  (base_seed={args.base_seed})")
    print(f"Iter budget   : {args.iter_budget}", end="")
    if args.algo == "nsga2":
        print(f"  → {gen} gen × {pop} pop = {gen * pop} evals")
    else:
        print()
    print()

    # Open CSV for streaming trial results
    with open(csv_path, "w", newline="") as csv_file:
        # CSV columns vary by algorithm — write dynamically after first trial
        writer_ref: list = [None]   # mutable container for the DictWriter

        def csv_callback(study, trial):
            if trial.value is None:
                return  # pruned or failed trial
            row = {"trial": trial.number, "value": round(trial.value, 6), **trial.params}
            if writer_ref[0] is None:
                writer_ref[0] = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                writer_ref[0].writeheader()
            writer_ref[0].writerow(row)
            csv_file.flush()

            best = study.best_value
            elapsed = time.time() - run_start
            print(
                f"  trial {trial.number:3d}/{args.trials}  "
                f"value={trial.value:.4f}  best={best:.4f}  "
                f"params={trial.params}  "
                f"elapsed={elapsed:.0f}s"
            )

        study_name = args.study_name or run_name
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=args.storage,
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=bool(args.storage),
        )

        if args.algo == "sa":
            obj = lambda trial: objective_sa(
                trial, args, evaluator, generate_random_map, improve_balance_spatial
            )
        else:
            obj = lambda trial: objective_nsga2(
                trial, args, evaluator, generate_random_map, nsga2_optimize
            )

        run_start = time.time()
        study.optimize(obj, n_trials=args.trials, callbacks=[csv_callback], n_jobs=1)

    # Write best_params.json
    best_params_path = run_dir / "best_params.json"
    best_params = {
        "algorithm":   args.algo,
        "best_value":  round(study.best_value, 6),
        "best_params": {k: round(v, 6) for k, v in study.best_params.items()},
        "n_trials":    args.trials,
        "eval_seeds":  args.eval_seeds,
        "iter_budget": args.iter_budget,
    }
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    total_time = time.time() - run_start
    print()
    print(f"Tuning complete in {total_time/60:.1f} min")
    print(f"Best value    : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    print(f"Best params   → {best_params_path}")
    print(f"Trial history → {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
