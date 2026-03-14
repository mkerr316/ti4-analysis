#!/usr/bin/env python3
"""
Monte Carlo benchmark for TI4 map optimization.

Supports two modes:
  (1) Main experiment: five-condition ablation (JFI-only, Moran-only, LSAP-only,
      JFI+Moran, full composite) via --conditions. Use --algorithms sa (default when
      --conditions is set). Output includes a condition column for analysis.
  (2) Methods justification: multi-algorithm, multi-budget runs (e.g. rs, hc, sa,
      sga, nsga2, ts) when --conditions is not set.

All algorithms receive an equal evaluation budget per budget level.
Supports multiprocessing via --workers (each seed is independent).

Usage:
    # Main experiment (four-condition ablation, SA)
    python scripts/benchmark_engine.py --conditions jfi_only,moran_only,lsap_only,full_composite \\
        --seeds 100 --budgets 10000,50000,100000,500000 --output-dir output/

    # Methods justification (algorithm comparison)
    python scripts/benchmark_engine.py --algorithms hc,sa,nsga2,ts [--budgets ...] [--output-dir PATH]

Outputs (inside --output-dir / benchmark_YYYYMMDD_HHMMSS/):
    results.csv      — one row per (seed, algorithm, budget[, condition])
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
    p = argparse.ArgumentParser(description="Monte Carlo benchmark: main experiment (--conditions) or methods justification (--algorithms)")
    p.add_argument("--seeds",       type=int, default=100,    help="Number of random seeds")
    p.add_argument("--hc-iter",     type=int, default=1000,   help="HC iterations per seed")
    p.add_argument("--sa-iter",     type=int, default=1000,   help="SA iterations per seed")
    p.add_argument("--nsga-gen",    type=int, default=50,     help="NSGA-II generations")
    p.add_argument("--nsga-pop",    type=int, default=20,     help="NSGA-II population size")
    p.add_argument("--base-seed",   type=int, default=0,      help="First random seed")
    p.add_argument("--players",     type=int, default=6,      help="Number of players")
    p.add_argument("--output-dir",  type=str, default="output", help="Root output directory")
    p.add_argument("--algorithms",  type=str, default="hc,sa,nsga2",
                   help="Comma-separated algorithms to run (hc, sa, sga, nsga2, ts, rs)")
    p.add_argument("--workers",     type=int, default=1,
                   help="Parallel workers (default: 1 = sequential)")
    p.add_argument("--ts-tenure",   type=int, default=None,
                   help="TS tabu tenure (default: max(3, ceil(0.05*C(S,2)))).")
    p.add_argument("--ts-k",        type=float, default=None,
                   help="TS tenure coefficient k: θ = max(3, ceil(k·√S)). Overrides default if set.")
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
    p.add_argument("--sga-blob",    type=float, default=0.5,
                   help="SGA blob_fraction (default: 0.5)")
    p.add_argument("--sga-mut",     type=float, default=0.05,
                   help="SGA mutation_rate (default: 0.05)")
    p.add_argument("--sga-warm",    type=float, default=0.10,
                   help="SGA warm_fraction (default: 0.10)")
    p.add_argument("--budgets",     type=str, default=None,
                   help="Comma-separated evaluation budgets for convergence profile "
                        "(e.g. 1000,5000,10000,25000).  When set, overrides "
                        "--hc-iter / --sa-iter / --nsga-gen for each budget level. "
                        "NSGA-II generations = budget // nsga-pop.")
    p.add_argument("--chains",     type=int, default=1,
                   help="Number of independent chains per (seed, algorithm, budget) for "
                        "R-hat convergence diagnostic (default: 1). Use 3+ for rigor.")
    p.add_argument("--conditions", type=str, default=None,
                   help="Comma-separated condition names for main experiment: "
                        "jfi_only,moran_only,lsap_only,jfi_moran,full_composite. When set, "
                        "algorithms default to sa and --weight-grid-step must be 0.")
    p.add_argument("--weight-grid-step", type=float, default=0.0,
                   help="Simplex step size for SO algorithm weight grid (default: 0 = "
                        "disabled). Incompatible with --conditions.")
    p.add_argument("--corrected-landscape", action="store_true",
                   help="Enable structural corrections: Gen-0 static normalization, "
                        "smooth objectives, sqrt(k_i) LSAP, TS tenure 0.05*C(S,2).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Condition weights (main experiment: five-condition ablation)
# ---------------------------------------------------------------------------

CONDITIONS = {
    "jfi_only":       (0.0, 1.0, 0.0),                      # C0: (morans_i, jains_index, lisa_penalty)
    "moran_only":     (1.0, 0.0, 0.0),                      # C1
    "lsap_only":      (0.0, 0.0, 1.0),                      # C2
    "jfi_moran":      (0.5, 0.5, 0.0),                      # C3: JFI + global Moran, no LSAP (parsimony test)
    "full_composite": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),  # C4
}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "seed", "algorithm", "budget", "chain_id", "weight_vector", "condition",
    "balance_gap", "morans_i", "jains_index", "jfi_resources", "jfi_influence",
    "lisa_penalty", "composite_score", "elapsed_sec", "front_size",
    "evals_to_best",
]


def make_row(
    seed: int,
    algorithm: str,
    score,
    elapsed: float,
    front_size: int,
    budget: int = 0,
    evals_to_best: int = -1,
    chain_id: int = 0,
    weight_vector: str = "none",
    condition: str = "none",
) -> Dict:
    return {
        "seed":           seed,
        "algorithm":      algorithm,
        "budget":         budget,
        "chain_id":       chain_id,
        "weight_vector":  weight_vector,
        "condition":      condition,
        "balance_gap":    round(float(score.balance_gap),        4),
        "morans_i":       round(float(score.morans_i),           4),
        "jains_index":    round(float(score.jains_index),        4),
        "jfi_resources":  round(float(score.jfi_resources),      4),
        "jfi_influence":  round(float(score.jfi_influence),      4),
        "lisa_penalty":   round(float(score.lisa_penalty),       4),
        "composite_score":round(float(score.composite_score()),  4),
        "elapsed_sec":    round(elapsed,                          2),
        "front_size":     front_size,
        "evals_to_best":  evals_to_best,
    }


def make_error_row(seed: int, algorithm: str, elapsed: float, budget: int = 0,
                   chain_id: int = 0, weight_vector: str = "none",
                   condition: str = "none") -> Dict:
    return {
        "seed": seed, "algorithm": algorithm, "budget": budget, "chain_id": chain_id,
        "weight_vector": weight_vector, "condition": condition,
        "balance_gap": float("nan"), "morans_i": float("nan"),
        "jains_index": float("nan"), "jfi_resources": float("nan"),
        "jfi_influence": float("nan"), "lisa_penalty": float("nan"),
        "composite_score": float("nan"),
        "elapsed_sec": round(elapsed, 2), "front_size": -1,
        "evals_to_best": -1,
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
    for algo in ("rs", "hc", "sa", "sga", "nsga2", "ts"):
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
# Weight grid helpers
# ---------------------------------------------------------------------------

def _simplex_weight_grid(step: float):
    """
    All (w_mi, w_jfi, w_lp) triples on the 3-simplex at the given step size,
    always including the canonical 5:5:3 weight.

    With step=0.2  → 21 vectors (C(7,2) on a 5-point simplex).
    With step=0.1  → 66 vectors.
    """
    vectors = set()
    n = round(1.0 / step)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            vectors.add((round(i / n, 8), round(j / n, 8), round(k / n, 8)))
    # Always include the canonical 5:5:3 weight
    vectors.add((round(5 / 13, 8), round(5 / 13, 8), round(3 / 13, 8)))
    return sorted(vectors)


def _wv_label(w_mi: float, w_jfi: float, w_lp: float) -> str:
    return f"{w_mi:.4f}:{w_jfi:.4f}:{w_lp:.4f}"


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


def _sample_trajectory(history, n_samples: int = 11):
    """Sample best-so-far composite at n_samples points (0%, 10%, ..., 100% of evals).
    history: list of (eval_count, score). Returns list of (eval_count, trajectory_value).
    """
    if not history:
        return []
    max_evals = history[-1][0]
    if max_evals <= 0:
        return [(0, float(history[0][1].composite_score()))]
    best_so_far = float("inf")
    result = []
    idx = 0
    for pct in [i / (n_samples - 1) for i in range(n_samples)]:
        target = int(pct * max_evals)
        while idx < len(history) and history[idx][0] <= target:
            best_so_far = min(best_so_far, float(history[idx][1].composite_score()))
            idx += 1
        if idx > 0:
            result.append((target, best_so_far))
        else:
            result.append((target, float(history[0][1].composite_score())))
    return result


def _run_seed(job):
    """Run all algorithms for one (seed, budget) pair.

    When chains > 1, runs each algorithm chains times with distinct run seeds and
    records trajectory per chain for R-hat. Takes a single tuple (for Pool.map).
    Returns (ok: bool, rows: list[dict], archives, hv_archives, trajectories).
    """
    global _evaluator
    (seed, algos_list, budget, hc_iter, sa_iter, nsga_gen,
     players, sa_rate, sa_min_temp, nsga_pop, nsga_blob,
     nsga_mut, nsga_warm, ts_tenure, ts_k,
     sga_blob, sga_mut, sga_warm, chains, corrected_landscape,
     weight_grid_step, conditions_list) = job

    algos = set(algos_list)
    n_chains = max(1, int(chains))

    # Weight vectors: from --conditions (main experiment) or simplex grid or single default.
    if conditions_list is not None:
        weight_vectors = [CONDITIONS[c] for c in conditions_list]
        condition_labels = conditions_list
    elif weight_grid_step > 0.0:
        weight_vectors = _simplex_weight_grid(weight_grid_step)
        condition_labels = None
    else:
        weight_vectors = [(round(1.0 / 3.0, 8), round(1.0 / 3.0, 8), round(1.0 / 3.0, 8))]
        condition_labels = None
    k_wv = len(weight_vectors)
    per_vector_budget = max(1, budget // k_wv)

    import numpy as np

    def _hv2d(pts2d, ref2):
        """Exact 2D hypervolume. pts2d shape (N,2), minimisation."""
        if len(pts2d) == 0:
            return 0.0
        # Keep non-dominated points (minimisation)
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

    def _hv3d(scores, ref, rng=None, n_samples=None):
        """
        Exact 3D Hypervolume via sweep-line (WFG/HSO-style).
        Replaces the previous Monte Carlo approximation.
        rng and n_samples are retained as unused kwargs for call-site compatibility.
        """
        if not scores:
            return 0.0
        # Use objective_values_for_pareto() so HV matches dominance (smooth/raw consistent)
        pts = np.array([s.objective_values_for_pareto() for s in scores])
        # Third objective is raw lisa_penalty; normalize to [0,1] for ref point
        n = max(1, scores[0].n_spatial) if scores else 1
        pts[:, 2] = pts[:, 2] / max(1, n * (n - 1))
        # Clip to reference point — points outside ref contribute nothing
        pts = pts[np.all(pts < ref, axis=1)]
        if len(pts) == 0:
            return 0.0

        # Sort by third objective ascending
        pts = pts[pts[:, 2].argsort()]

        hv = 0.0
        prev_z = ref[2]
        # Sweep from largest z to smallest, maintaining 2D front at each slice
        for i in range(len(pts) - 1, -1, -1):
            slice_depth = prev_z - pts[i][2]
            if slice_depth <= 0:
                prev_z = pts[i][2]
                continue
            # Compute 2D hypervolume of current front projected onto xy plane
            front_2d = pts[: i + 1]  # all points with z <= pts[i][2] already processed
            hv2 = _hv2d(front_2d[:, :2], ref[:2])
            hv += hv2 * slice_depth
            prev_z = pts[i][2]
        return hv

    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.hc_optimizer import hc_optimize
    from ti4_analysis.algorithms.spatial_optimizer import (
        improve_balance_spatial, evaluate_map_multiobjective,
        compute_gen0_sigma,
    )
    from ti4_analysis.algorithms.nsga2_optimizer import nsga2_optimize
    from ti4_analysis.algorithms.sga_optimizer import sga_optimize
    from ti4_analysis.algorithms.tabu_search_optimizer import improve_balance_tabu
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState

    evaluator = _evaluator
    rows = []
    _pareto_archives = []
    _hv_archives = []  # (algorithm, seed, budget, chain_id, rows)
    _trajectories = []  # (algorithm, seed, budget, chain_id, [(eval_count, value), ...])
    seed_start = time.time()

    try:
        initial_map = generate_random_map(
            player_count=players,
            template_name="normal",
            include_pok=True,
            random_seed=seed,
        )

        # Structural corrections: Gen-0 sigma + smooth + local-variance LSAP (frozen for run)
        normalizer_sigma = None
        eval_kw = {}
        if corrected_landscape:
            topo_gen0 = MapTopology.from_ti4_map(initial_map, evaluator)
            normalizer_sigma = compute_gen0_sigma(
                topo_gen0, evaluator, initial_map.copy(),
                n_samples=1000, random_seed=seed + 99999,
                use_local_variance_lisa=True,
            )
            eval_kw = dict(
                normalizer_sigma=normalizer_sigma,
                use_smooth_objectives=True,
                smooth_p=8.0,
                smooth_k=10.0,
                use_local_variance_lisa=True,
            )

        # Helper: incremental empirical Pareto front over MultiObjectiveScore
        def _update_front(front, score):
            """
            Maintain a non-dominated set over canonical objectives:
            1-JFI, |Moran's I|, LSAP (all minimised).
            """
            # Discard if dominated by an existing point (or effectively identical)
            for f in front:
                if f.dominates(score):
                    return front
                if (math.isclose(f.jains_index, score.jains_index, rel_tol=1e-6) and
                        math.isclose(f.morans_i, score.morans_i, rel_tol=1e-6) and
                        math.isclose(f.lisa_penalty, score.lisa_penalty, rel_tol=1e-6)):
                    return front

            # Remove any points that are dominated by the new score
            new_front = [f for f in front if not score.dominates(f)]
            new_front.append(score)
            return new_front

        def _front_to_rows(front):
            """Convert a list of MultiObjectiveScore to archive rows."""
            rows_local = []
            for s in front:
                rows_local.append({
                    "jains_index": round(float(s.jains_index), 6),
                    "morans_i": round(float(s.morans_i), 6),
                    "lisa_penalty": round(float(s.lisa_penalty), 6),
                })
            return rows_local

        # ── Random Search (baseline) ──────────────────────────────
        if "rs" in algos:
            for chain_id in range(n_chains):
                rs_agg_front = []
                rs_history_canonical = []
                for wv_idx, (w_mi, w_jfi, w_lp) in enumerate(weight_vectors):
                    weights = {"morans_i": w_mi, "jains_index": w_jfi, "lisa_penalty": w_lp}
                    wv_kw = {**eval_kw, "weights": weights}
                    wv_label = condition_labels[wv_idx] if condition_labels else (_wv_label(w_mi, w_jfi, w_lp) if k_wv > 1 else "none")
                    condition = condition_labels[wv_idx] if condition_labels else "none"
                    # Preserve original seed formula when k==1 for reproducibility
                    if k_wv == 1:
                        wv_run_seed = seed * 1000 + chain_id
                    else:
                        wv_run_seed = seed * 100_000 + chain_id * 1000 + wv_idx
                    rs_map = initial_map.copy()
                    t0 = time.time()
                    topo = MapTopology.from_ti4_map(rs_map, evaluator)
                    fs_base = FastMapState.from_ti4_map(topo, rs_map, evaluator)
                    best_rs_score = evaluate_map_multiobjective(
                        rs_map, evaluator, fast_state=fs_base, **wv_kw
                    )
                    rs_front = [best_rs_score]
                    rng_rs = np.random.default_rng(wv_run_seed)
                    S = len(topo.swappable_indices)
                    rs_etb = 0
                    rs_history = [(0, best_rs_score)]
                    for ev in range(1, per_vector_budget):
                        fs = fs_base.clone()
                        perm = rng_rs.permutation(S)
                        for i in range(S - 1, 0, -1):
                            if perm[i] != i:
                                fs.swap(perm[i], i)
                        candidate = evaluate_map_multiobjective(
                            rs_map, evaluator, fast_state=fs, **wv_kw
                        )
                        rs_front = _update_front(rs_front, candidate)
                        if candidate.composite_score() < best_rs_score.composite_score():
                            best_rs_score = candidate
                            rs_etb = ev
                        rs_history.append((ev, best_rs_score))
                        del fs
                    rows.append(make_row(seed, "rs", best_rs_score, time.time() - t0, 1, budget,
                                         evals_to_best=rs_etb, chain_id=chain_id,
                                         weight_vector=wv_label, condition=condition))
                    for sc in rs_front:
                        rs_agg_front = _update_front(rs_agg_front, sc)
                    if wv_idx == 0:
                        rs_history_canonical = rs_history
                    del rs_map, topo, fs_base
                _hv_archives.append(("rs", seed, budget, chain_id, _front_to_rows(rs_agg_front)))
                if n_chains > 1 and k_wv == 1:
                    _trajectories.append(("rs", seed, budget, chain_id,
                                          _sample_trajectory(rs_history_canonical)))

        if "hc" in algos:
            for chain_id in range(n_chains):
                hc_agg_front = []
                hc_history_canonical = []
                for wv_idx, (w_mi, w_jfi, w_lp) in enumerate(weight_vectors):
                    weights = {"morans_i": w_mi, "jains_index": w_jfi, "lisa_penalty": w_lp}
                    wv_label = condition_labels[wv_idx] if condition_labels else (_wv_label(w_mi, w_jfi, w_lp) if k_wv > 1 else "none")
                    condition = condition_labels[wv_idx] if condition_labels else "none"
                    if k_wv == 1:
                        wv_run_seed = seed * 1000 + chain_id
                    else:
                        wv_run_seed = seed * 100_000 + chain_id * 1000 + wv_idx
                    hc_map = initial_map.copy()
                    t0 = time.time()
                    hc_score, hc_history, hc_etb = hc_optimize(
                        hc_map, evaluator,
                        iterations=per_vector_budget,
                        random_seed=wv_run_seed,
                        verbose=False,
                        weights=weights,
                        **eval_kw,
                    )
                    rows.append(make_row(seed, "hc", hc_score, time.time() - t0, 1, budget,
                                         evals_to_best=hc_etb, chain_id=chain_id,
                                         weight_vector=wv_label, condition=condition))
                    for _, sc in hc_history:
                        hc_agg_front = _update_front(hc_agg_front, sc)
                    if wv_idx == 0:
                        hc_history_canonical = hc_history
                    del hc_map
                _hv_archives.append(("hc", seed, budget, chain_id, _front_to_rows(hc_agg_front)))
                if n_chains > 1 and k_wv == 1:
                    _trajectories.append(("hc", seed, budget, chain_id,
                                          _sample_trajectory(hc_history_canonical)))

        if "sa" in algos:
            for chain_id in range(n_chains):
                sa_agg_front = []
                sa_history_canonical = []
                for wv_idx, (w_mi, w_jfi, w_lp) in enumerate(weight_vectors):
                    weights = {"morans_i": w_mi, "jains_index": w_jfi, "lisa_penalty": w_lp}
                    wv_label = condition_labels[wv_idx] if condition_labels else (_wv_label(w_mi, w_jfi, w_lp) if k_wv > 1 else "none")
                    condition = condition_labels[wv_idx] if condition_labels else "none"
                    if k_wv == 1:
                        wv_run_seed = seed * 1000 + chain_id
                    else:
                        wv_run_seed = seed * 100_000 + chain_id * 1000 + wv_idx
                    sa_map = initial_map.copy()
                    t0 = time.time()
                    sa_score, sa_history, sa_etb = improve_balance_spatial(
                        sa_map, evaluator,
                        iterations=per_vector_budget,
                        initial_acceptance_rate=sa_rate,
                        min_temp=sa_min_temp,
                        random_seed=wv_run_seed,
                        verbose=False,
                        weights=weights,
                        **eval_kw,
                    )
                    rows.append(make_row(seed, "sa", sa_score, time.time() - t0, 1, budget,
                                         evals_to_best=sa_etb, chain_id=chain_id,
                                         weight_vector=wv_label, condition=condition))
                    for _, sc in sa_history:
                        sa_agg_front = _update_front(sa_agg_front, sc)
                    if wv_idx == 0:
                        sa_history_canonical = sa_history
                    del sa_map
                _hv_archives.append(("sa", seed, budget, chain_id, _front_to_rows(sa_agg_front)))
                if n_chains > 1 and k_wv == 1:
                    _trajectories.append(("sa", seed, budget, chain_id,
                                          _sample_trajectory(sa_history_canonical)))

        if "nsga2" in algos:
            ref_hv = np.array([1.2, 1.2, 1.2])
            for chain_id in range(n_chains):
                run_seed = seed * 1000 + chain_id
                nsga_traj = []

                def _nsga_cb(gen, rank0_scores):
                    hv = _hv3d(rank0_scores, ref_hv)
                    nsga_traj.append((gen * nsga_pop, hv))

                nsga_map = initial_map.copy()
                t0 = time.time()
                front = nsga2_optimize(
                    nsga_map, evaluator,
                    generations=nsga_gen,
                    population_size=nsga_pop,
                    blob_fraction=nsga_blob,
                    mutation_rate=nsga_mut,
                    warm_fraction=nsga_warm,
                    random_seed=run_seed,
                    verbose=False,
                    trajectory_callback=_nsga_cb,
                    use_smooth_objectives=eval_kw.get("use_smooth_objectives", False),
                    smooth_p=eval_kw.get("smooth_p", 8.0),
                    smooth_k=eval_kw.get("smooth_k", 10.0),
                    use_local_variance_lisa=eval_kw.get("use_local_variance_lisa", False),
                    normalizer_sigma=eval_kw.get("normalizer_sigma"),
                )
                best_score = min(front, key=lambda x: x[1].composite_score())[1]
                rows.append(make_row(seed, "nsga2", best_score, time.time() - t0,
                                     len(front), budget, chain_id=chain_id, condition="none"))

                # Save Pareto archive for Track B quality indicators
                archive_rows = []
                hv_rows = []
                for _, s in front:
                    row_jfi = round(float(s.jains_index), 6)
                    row_mi = round(float(s.morans_i), 6)
                    row_lp = round(float(s.lisa_penalty), 6)
                    archive_rows.append({
                        "jains_index": row_jfi,
                        "morans_i": row_mi,
                        "lisa_penalty": row_lp,
                        "composite_score": round(float(s.composite_score()), 6),
                    })
                    hv_rows.append({
                        "jains_index": row_jfi,
                        "morans_i": row_mi,
                        "lisa_penalty": row_lp,
                    })
                if chain_id == 0:
                    _pareto_archives.append((seed, budget, archive_rows))
                _hv_archives.append(("nsga2", seed, budget, chain_id, hv_rows))
                if n_chains > 1 and nsga_traj:
                    _trajectories.append(("nsga2", seed, budget, chain_id, nsga_traj))
                del nsga_map, front

        if "sga" in algos:
            for chain_id in range(n_chains):
                sga_agg_front = []
                sga_history_canonical = []
                for wv_idx, (w_mi, w_jfi, w_lp) in enumerate(weight_vectors):
                    weights = {"morans_i": w_mi, "jains_index": w_jfi, "lisa_penalty": w_lp}
                    wv_label = condition_labels[wv_idx] if condition_labels else (_wv_label(w_mi, w_jfi, w_lp) if k_wv > 1 else "none")
                    condition = condition_labels[wv_idx] if condition_labels else "none"
                    if k_wv == 1:
                        wv_run_seed = seed * 1000 + chain_id
                    else:
                        wv_run_seed = seed * 100_000 + chain_id * 1000 + wv_idx
                    sga_map = initial_map.copy()
                    t0 = time.time()
                    sga_gen = max(1, per_vector_budget // nsga_pop)
                    sga_score, sga_history = sga_optimize(
                        sga_map, evaluator,
                        generations=sga_gen,
                        population_size=nsga_pop,
                        blob_fraction=sga_blob,
                        mutation_rate=sga_mut,
                        warm_fraction=sga_warm,
                        random_seed=wv_run_seed,
                        verbose=False,
                        weights=weights,
                        **eval_kw,
                    )
                    sga_etb = 0
                    best_c = float("inf")
                    for evals, sc in sga_history:
                        if sc.composite_score() < best_c:
                            best_c = sc.composite_score()
                            sga_etb = evals
                        sga_agg_front = _update_front(sga_agg_front, sc)
                    rows.append(make_row(seed, "sga", sga_score, time.time() - t0, 1, budget,
                                         evals_to_best=sga_etb, chain_id=chain_id,
                                         weight_vector=wv_label, condition=condition))
                    if wv_idx == 0:
                        sga_history_canonical = sga_history
                    del sga_map
                _hv_archives.append(("sga", seed, budget, chain_id, _front_to_rows(sga_agg_front)))
                if n_chains > 1 and k_wv == 1:
                    _trajectories.append(("sga", seed, budget, chain_id,
                                          _sample_trajectory(sga_history_canonical)))

        if "ts" in algos:
            for chain_id in range(n_chains):
                ts_agg_front = []
                ts_history_canonical = []
                for wv_idx, (w_mi, w_jfi, w_lp) in enumerate(weight_vectors):
                    weights = {"morans_i": w_mi, "jains_index": w_jfi, "lisa_penalty": w_lp}
                    wv_label = condition_labels[wv_idx] if condition_labels else (_wv_label(w_mi, w_jfi, w_lp) if k_wv > 1 else "none")
                    condition = condition_labels[wv_idx] if condition_labels else "none"
                    if k_wv == 1:
                        wv_run_seed = seed * 1000 + chain_id
                    else:
                        wv_run_seed = seed * 100_000 + chain_id * 1000 + wv_idx
                    ts_map = initial_map.copy()
                    t0 = time.time()
                    ts_kw = dict(**eval_kw)
                    ts_kw["weights"] = weights
                    if ts_k is not None:
                        ts_kw["tabu_tenure_coefficient"] = ts_k
                    elif ts_tenure is not None:
                        ts_kw["tabu_tenure"] = ts_tenure
                    ts_score, ts_history, ts_etb, _ = improve_balance_tabu(
                        ts_map, evaluator,
                        max_evaluations=per_vector_budget,
                        random_seed=wv_run_seed,
                        verbose=False,
                        **ts_kw,
                    )
                    rows.append(make_row(seed, "ts", ts_score, time.time() - t0, 1, budget,
                                         evals_to_best=ts_etb, chain_id=chain_id,
                                         weight_vector=wv_label, condition=condition))
                    for _, sc in ts_history:
                        ts_agg_front = _update_front(ts_agg_front, sc)
                    if wv_idx == 0:
                        ts_history_canonical = ts_history
                    del ts_map
                _hv_archives.append(("ts", seed, budget, chain_id, _front_to_rows(ts_agg_front)))
                if n_chains > 1 and k_wv == 1:
                    _trajectories.append(("ts", seed, budget, chain_id,
                                          _sample_trajectory(ts_history_canonical)))

        return True, rows, _pareto_archives, _hv_archives, _trajectories

    except Exception as exc:
        elapsed_err = time.time() - seed_start
        print(f"seed={seed:4d}  ERROR after {elapsed_err:.1f}s: {exc}", file=sys.stderr)
        error_rows = []
        for algo in algos:
            for chain_id in range(n_chains):
                error_rows.append(make_error_row(seed, algo, elapsed_err, budget, chain_id))
        return False, error_rows, [], [], []

    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Mutual exclusivity: --conditions and --weight-grid-step > 0 cannot both be set.
    weight_grid_step = getattr(args, "weight_grid_step", 0.0)
    if getattr(args, "conditions", None) and weight_grid_step > 0.0:
        print("ERROR: Cannot use --conditions with --weight-grid-step > 0; "
              "they specify objective weights in incompatible ways.", file=sys.stderr)
        return 1

    # When running main experiment (--conditions), default to SA only.
    if getattr(args, "conditions", None):
        args.algorithms = "sa"
    algos = {a.strip().lower() for a in args.algorithms.split(",")}
    valid = {"hc", "sa", "sga", "nsga2", "ts", "rs"}
    unknown = algos - valid
    if unknown:
        print(f"ERROR: Unknown algorithms: {unknown}. Valid: {valid}", file=sys.stderr)
        return 1

    # Parse and validate condition names for main experiment.
    conditions_list = None
    if getattr(args, "conditions", None):
        conditions_list = [c.strip() for c in args.conditions.split(",") if c.strip()]
        invalid = [c for c in conditions_list if c not in CONDITIONS]
        if invalid:
            print(f"ERROR: Unknown conditions: {invalid}. Valid: {list(CONDITIONS)}", file=sys.stderr)
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
        "ts_tenure":   args.ts_tenure,
        "ts_k":        args.ts_k,
        "chains":      max(1, getattr(args, "chains", 1)),
        "corrected_landscape": getattr(args, "corrected_landscape", False),
        "use_local_variance_lisa": getattr(args, "corrected_landscape", False),
        "weight_grid_step": getattr(args, "weight_grid_step", 0.0),
        "conditions": conditions_list,
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
    if conditions_list:
        print(f"Conditions    : {', '.join(conditions_list)}  ({len(conditions_list)} rows per seed)")
    print(f"Budget levels : {budget_levels}")
    print(f"Chains        : {max(1, getattr(args, 'chains', 1))}")
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

                n_chains = max(1, getattr(args, "chains", 1))
                weight_grid_step = getattr(args, "weight_grid_step", 0.0)
                if weight_grid_step > 0.0:
                    wg = _simplex_weight_grid(weight_grid_step)
                    k_log = len(wg)
                    pvb_log = max(1, budget // k_log)
                    print(f"   SO weight grid: step={weight_grid_step}, k={k_log} vectors, "
                          f"{pvb_log} evals/vector (total budget {budget} split equally)")

                jobs = [
                    (seed, sorted(algos), budget, hc_iter, sa_iter, nsga_gen,
                     args.players, args.sa_rate, args.sa_min_temp, args.nsga_pop,
                     args.nsga_blob, args.nsga_mut, args.nsga_warm, args.ts_tenure, args.ts_k,
                     args.sga_blob, args.sga_mut, args.sga_warm, n_chains,
                     getattr(args, "corrected_landscape", False),
                     weight_grid_step, conditions_list)
                    for seed in range(args.base_seed, args.base_seed + args.seeds)
                ]

                budget_done = 0
                budget_start = time.time()
                mapper = pool.imap_unordered if pool else map

                pareto_archive_dir = run_dir / "pareto_archives"
                if "nsga2" in algos:
                    pareto_archive_dir.mkdir(parents=True, exist_ok=True)
                hv_archive_dir = run_dir / "unified_archives"
                hv_archive_dir.mkdir(parents=True, exist_ok=True)
                traj_dir = run_dir / "trajectories"
                if n_chains > 1:
                    traj_dir.mkdir(parents=True, exist_ok=True)

                for ok, rows, archives, hv_archives, trajectories in mapper(_run_seed, jobs):
                    for row in rows:
                        writer.writerow(row)
                    csv_file.flush()

                    for (a_seed, a_budget, a_rows) in archives:
                        a_path = pareto_archive_dir / f"pareto_archive_seed{a_seed}_b{a_budget}.csv"
                        with open(a_path, "w", newline="") as af:
                            aw = csv.DictWriter(af, fieldnames=["jains_index", "morans_i",
                                                                 "lisa_penalty", "composite_score"])
                            aw.writeheader()
                            for ar in a_rows:
                                aw.writerow(ar)

                    # Unified empirical fronts (all algorithms); one file per chain when chains > 1
                    for entry in hv_archives:
                        algo, h_seed, h_budget, h_chain_id, h_rows = entry
                        chain_suffix = f"_chain{h_chain_id}" if n_chains > 1 else ""
                        h_path = hv_archive_dir / f"unified_archive_algo{algo}_seed{h_seed}_b{h_budget}{chain_suffix}.csv"
                        with open(h_path, "w", newline="") as hf:
                            hw = csv.DictWriter(hf, fieldnames=["jains_index", "morans_i",
                                                                "lisa_penalty"])
                            hw.writeheader()
                            for hr in h_rows:
                                hw.writerow(hr)

                    # Trajectories for R-hat: one file per (algo, seed, budget) with all chains
                    traj_by_key = defaultdict(list)  # (algo, seed, budget) -> [(chain_id, eval_count, value), ...]
                    for algo, t_seed, t_budget, t_chain_id, traj_list in trajectories:
                        key = (algo, t_seed, t_budget)
                        for ev, val in traj_list:
                            traj_by_key[key].append((t_chain_id, ev, round(val, 6)))
                    for (algo, t_seed, t_budget), points in traj_by_key.items():
                        t_path = traj_dir / f"trajectories_algo{algo}_seed{t_seed}_b{t_budget}.csv"
                        with open(t_path, "w", newline="") as tf:
                            tw = csv.DictWriter(tf, fieldnames=["chain_id", "eval_count", "trajectory_value"])
                            tw.writeheader()
                            for chain_id, ev, val in sorted(points, key=lambda x: (x[0], x[1])):
                                tw.writerow({"chain_id": chain_id, "eval_count": ev, "trajectory_value": val})

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
                    for a in ("rs", "hc", "sa", "sga", "nsga2", "ts"):
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
