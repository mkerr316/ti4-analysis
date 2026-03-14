"""
Deterministic hill-climber on the composite score (5:5:3 JFI + Moran's I + LSAP).

Used by the benchmark pipeline so that the "HC" baseline optimizes the same
objective as SA, SGA, and TS. Accepts a swap if and only if it strictly
reduces the composite score (no temperature, no Metropolis criterion).

The gap-only HC in balance_engine.improve_balance is retained for the
spatial blindness experiment and for warm-starting NSGA-II/SGA.
"""

import random
from typing import Tuple, List, Optional, Dict

from .balance_engine import TI4Map
from .map_topology import MapTopology
from .fast_map_state import FastMapState
from .spatial_optimizer import (
    MultiObjectiveScore,
    evaluate_map_multiobjective,
)
from ..data.map_structures import Evaluator


def hc_optimize(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    iterations: int = 200,
    weights: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
    verbose: bool = False,
    normalizer_sigma: Optional[Dict[str, float]] = None,
    use_smooth_objectives: bool = False,
    smooth_p: float = 8.0,
    smooth_k: float = 10.0,
    use_local_variance_lisa: bool = False,
) -> Tuple[MultiObjectiveScore, List[Tuple[int, MultiObjectiveScore]], int]:
    """
    Hill-climb on the composite score (strict improvement only).

    Same swap loop structure as SA but with no temperature: accept iff
    delta < 0. Uses MapTopology and FastMapState for O(1) swap and
    vectorized evaluation.

    Args:
        ti4_map: Map to optimize (modified in-place).
        evaluator: Balance evaluator.
        iterations: Maximum number of swap attempts.
        weights: Objective weights (default 5:5:3).
        random_seed: Random seed for reproducibility.
        verbose: Print progress messages.

    Returns:
        (best_score, history, evals_to_best)
        - best_score: MultiObjectiveScore at the best iteration.
        - history: List of (iteration, score) every 10 iterations plus final.
        - evals_to_best: Iteration at which the incumbent was last improved.
    """
    if random_seed is not None:
        random.seed(random_seed)

    topology = MapTopology.from_ti4_map(ti4_map, evaluator)
    fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)
    swappable_spaces = [ti4_map.spaces[i] for i in topology.swappable_indices]
    sample_indices = list(range(len(swappable_spaces)))

    eval_kw = dict(
        normalizer_sigma=normalizer_sigma,
        use_smooth_objectives=use_smooth_objectives,
        smooth_p=smooth_p,
        smooth_k=smooth_k,
        use_local_variance_lisa=use_local_variance_lisa,
    )
    if len(swappable_spaces) < 2:
        score = evaluate_map_multiobjective(ti4_map, evaluator, weights, fast_state, **eval_kw)
        return score, [(0, score)], 0

    current_score = evaluate_map_multiobjective(
        ti4_map, evaluator, weights, fast_state, **eval_kw
    )
    best_score = current_score
    best_eval = 0
    history = [(0, current_score)]

    for i in range(1, iterations + 1):
        s1, s2 = random.sample(sample_indices, 2)
        space1 = swappable_spaces[s1]
        space2 = swappable_spaces[s2]

        fast_state.swap(s1, s2)
        space1.system, space2.system = space2.system, space1.system

        new_score = evaluate_map_multiobjective(
            ti4_map, evaluator, weights, fast_state, **eval_kw
        )

        if new_score.lex_key() < current_score.lex_key():
            current_score = new_score
            if new_score.lex_key() < best_score.lex_key():
                best_score = new_score
                best_eval = i
                if verbose and i % 10 == 0:
                    print(f"Iteration {i}: {new_score}")
        else:
            fast_state.swap(s1, s2)
            space1.system, space2.system = space2.system, space1.system

        if i % 10 == 0:
            history.append((i, current_score))

    history.append((iterations, current_score))
    return best_score, history, best_eval
