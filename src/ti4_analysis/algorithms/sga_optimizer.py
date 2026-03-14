"""
Single-Objective Genetic Algorithm (SGA) for TI4 map balance.

Optimizes the same 1:1:1 composite scalar that SA hill-climbs, using the
same population-based operators (BFS-blob OX1 crossover, swap mutation,
inoculation seeding) as NSGA-II. This isolates the comparison:

    SA vs SGA   → Markov chain vs population-based, same scalar objective
    SGA vs NSGA-II → single-objective vs multi-objective, same operators
    SA vs NSGA-II  → both architecture and objective differ

Selection uses scalar binary tournament on composite_score() (lower = better).
Replacement is (μ + λ) elitist truncation: the N lowest-composite individuals
from the 2N parent+offspring pool survive. No Pareto sorting, no crowding
distance — zero multi-objective overhead.
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .balance_engine import TI4Map, improve_balance
from .map_topology import MapTopology
from .fast_map_state import FastMapState
from ..data.map_structures import Evaluator
from .spatial_optimizer import MultiObjectiveScore, evaluate_map_multiobjective
from .nsga2_optimizer import (
    _build_lookups,
    _bfs_blob,
    _ox1_crossover,
    _build_offspring,
    _seed_population,
    Individual,
)


# ── Scalar tournament selection ───────────────────────────────────────────────

def _scalar_tournament(pool: List[Individual], rng: random.Random) -> Individual:
    """Binary tournament on lex_key (lower wins)."""
    a, b = rng.sample(pool, 2)
    return a if a.score.lex_key() <= b.score.lex_key() else b


# ── Main SGA loop ─────────────────────────────────────────────────────────────

def sga_optimize(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    generations: int = 200,
    population_size: int = 100,
    blob_fraction: float = 0.5,
    mutation_rate: float = 0.05,
    warm_fraction: float = 0.10,
    random_seed: Optional[int] = None,
    verbose: bool = True,
    weights: Optional[dict] = None,
    normalizer_sigma: Optional[dict] = None,
    use_smooth_objectives: bool = False,
    smooth_p: float = 8.0,
    smooth_k: float = 10.0,
    use_local_variance_lisa: bool = False,
) -> Tuple[MultiObjectiveScore, List[Tuple[int, MultiObjectiveScore]]]:
    """
    Single-objective GA with BFS-blob OX1 crossover for TI4 map balance.

    Optimises the weighted composite scalar (1:1:1 by default) that SA also
    optimises, using the same crossover and mutation operators as NSGA-II.

    Args:
        ti4_map        : Starting map (defines tile pool and board topology)
        evaluator      : Balance evaluator (resource/influence weights)
        generations    : Number of evolutionary generations
        population_size: N individuals per generation (2N evaluated per gen)
        blob_fraction  : Fraction of swappable positions per BFS crossover blob
        mutation_rate  : Per-position swap probability after crossover
        warm_fraction  : Fraction of Gen 0 seeded with HC-optimised maps
        random_seed    : Reproducibility seed (None = non-deterministic)
        verbose        : Print per-generation progress
        weights        : Objective weights dict (morans_i, jains_index,
                         lisa_penalty). Defaults to 1:1:1 if None.

    Returns:
        Tuple of (best_score, history) where history is a list of
        (evaluation_count, best_score_at_that_point) tuples.
    """
    rng = random.Random(random_seed)

    topology = MapTopology.from_ti4_map(ti4_map, evaluator)
    _, spatial_row_to_s_pos, s_pos_to_spatial_row = _build_lookups(topology)
    S = len(topology.swappable_indices)

    if S < 2:
        raise ValueError("Not enough swappable spaces for SGA (need >= 2)")

    eval_kw = dict(
        weights=weights,
        normalizer_sigma=normalizer_sigma,
        use_smooth_objectives=use_smooth_objectives,
        smooth_p=smooth_p,
        smooth_k=smooth_k,
        use_local_variance_lisa=use_local_variance_lisa,
    )
    # ── Generation 0: seeding (shared with NSGA-II) ───────────────────────
    population = _seed_population(
        ti4_map, topology, evaluator, population_size, warm_fraction, rng, verbose,
        seed_eval_kwargs=eval_kw,
    )

    # Patch weights onto seed-population scores so composite_score() and
    # lex_key() use the caller-supplied weights from the first generation on.
    if weights is not None:
        for ind in population:
            ind.score.weights = weights

    total_evals = population_size
    best_ind = min(population, key=lambda ind: ind.score.lex_key())
    best_score = best_ind.score
    best_eval = total_evals
    history: List[Tuple[int, MultiObjectiveScore]] = [(total_evals, best_score)]

    if verbose:
        print(f"Gen 0 seed complete: best composite={best_score.composite_score():.4f}")

    # ── Evolutionary loop ─────────────────────────────────────────────────
    for gen in range(1, generations + 1):

        offspring: List[Individual] = []
        while len(offspring) < population_size:
            pa = _scalar_tournament(population, rng)
            pb = _scalar_tournament(population, rng)

            pa_systems = [pa.map.spaces[i].system for i in topology.swappable_indices]
            pb_systems = [pb.map.spaces[i].system for i in topology.swappable_indices]

            blob = _bfs_blob(
                topology, s_pos_to_spatial_row, spatial_row_to_s_pos, rng, blob_fraction
            )
            child_systems = _ox1_crossover(pa_systems, pb_systems, blob)

            for s in range(S):
                if rng.random() < mutation_rate:
                    s2 = rng.randrange(S)
                    if s2 != s:
                        child_systems[s], child_systems[s2] = (
                            child_systems[s2], child_systems[s]
                        )

            child_map, child_state = _build_offspring(
                pa.map, topology, child_systems, evaluator
            )
            child_score = evaluate_map_multiobjective(
                child_map, evaluator, fast_state=child_state, **eval_kw
            )
            offspring.append(Individual(
                map=child_map, fast_state=child_state, score=child_score
            ))

        total_evals += len(offspring)

        # (μ + λ) elitist truncation on lex_key
        combined = population + offspring
        combined.sort(key=lambda ind: ind.score.lex_key())
        population = combined[:population_size]

        gen_best = population[0].score
        if gen_best.lex_key() < best_score.lex_key():
            best_score = gen_best
            best_eval = total_evals

        if verbose and (gen % 10 == 0 or gen == generations):
            print(
                f"Gen {gen:>{len(str(generations))}}/{generations}: "
                f"best={best_score.composite_score():.4f} | "
                f"gen_best={gen_best.composite_score():.4f} | "
                f"evals={total_evals}"
            )

        history.append((total_evals, best_score))

    if verbose:
        print(f"\nSGA complete: best composite={best_score.composite_score():.4f} "
              f"found at eval {best_eval}/{total_evals}")

    return best_score, history
