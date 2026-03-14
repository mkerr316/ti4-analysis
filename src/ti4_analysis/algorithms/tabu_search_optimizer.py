"""
Tabu Search optimizer for TI4 map balance.

Classic full-neighbourhood Tabu Search: each iteration evaluates all C(S,2)
possible 2-swaps among S swappable positions, selects the best non-tabu move
(or the best tabu move when the aspiration criterion is satisfied), and
unconditionally applies it.

The key distinction from Simulated Annealing:
    SA escapes local optima *stochastically* (Metropolis criterion);
    TS escapes them *deterministically* via short-term memory (tabu list).

Aspiration criterion: a tabu move is accepted if it produces a composite score
strictly better than the best-ever observed, overriding the tabu restriction.

Evaluation budget accounting: each candidate swap costs one evaluation. A single
TS iteration costs C(S,2) evaluations, making budget comparisons with SA/HC
evaluation-for-evaluation fair. At low budgets, TS completes fewer sequential
decisions (iterations) than SA — the depth-vs-breadth tradeoff mirrors
the NSGA-II population overhead argument.
"""

import random
from itertools import combinations
from typing import Tuple, List, Optional, Dict

from .balance_engine import TI4Map
from .map_topology import MapTopology
from .fast_map_state import FastMapState
from ..data.map_structures import Evaluator
from .spatial_optimizer import MultiObjectiveScore, evaluate_map_multiobjective


def improve_balance_tabu(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    max_evaluations: int = 1000,
    weights: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True,
    tabu_tenure: Optional[int] = None,
    tabu_tenure_coefficient: Optional[float] = None,
    tabu_tenure_neighborhood_fraction: Optional[float] = None,
    use_attribute_tabu: bool = False,
    stagnation_threshold: Optional[int] = None,
    normalizer_sigma: Optional[Dict[str, float]] = None,
    use_smooth_objectives: bool = False,
    smooth_p: float = 8.0,
    smooth_k: float = 10.0,
    use_local_variance_lisa: bool = True,
) -> Tuple[MultiObjectiveScore, List[Tuple[int, MultiObjectiveScore]], int, int]:
    """
    Improve map balance using Tabu Search over the composite fitness.

    Each iteration scans the full 2-swap neighbourhood, applies the best
    non-tabu move (or aspirational tabu move), and adds the reverse swap
    to the tabu list for `tabu_tenure` iterations.

    Tenure is resolved as: θ = max(3, ⌈k·√S⌉). If tabu_tenure (int) is given
    it is used directly; else if tabu_tenure_coefficient (float) is given
    then tenure = max(3, ceil(k * sqrt(S))); else default k=1 (Glover 1989).

    Args:
        ti4_map: Map to optimize (modified in-place)
        evaluator: Balance evaluator
        max_evaluations: Total evaluation budget (comparable to SA iterations)
        weights: Objective weights (morans_i, jains_index, lisa_penalty; sum to 1.0)
        random_seed: Random seed for reproducibility
        verbose: Print progress messages
        tabu_tenure: Number of iterations a swap remains forbidden (fixed value).
        tabu_tenure_coefficient: If set, tenure = max(3, ceil(k * sqrt(S))). Ignored if tabu_tenure is set.
        use_attribute_tabu: If True, also tabu (position, system_id) assignments for tenure steps.
        stagnation_threshold: If set, after this many iterations without improvement, apply random
            diversification swaps and reset. Typical: X = max(1, int(0.1 * C(S, 2))). If None, disabled.

    Returns:
        Tuple of (best_score, history, evals_to_best, iterations_completed)
        history is a list of (evaluation_count, score) tuples
        evals_to_best is the evaluation count when the incumbent was last improved
        iterations_completed is the number of TS iterations (full or partial) executed
    """
    import math as _math

    if random_seed is not None:
        random.seed(random_seed)

    topology = MapTopology.from_ti4_map(ti4_map, evaluator)
    fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)
    swappable_spaces = [ti4_map.spaces[i] for i in topology.swappable_indices]
    S = len(swappable_spaces)

    if S < 2:
        raise ValueError("Not enough swappable spaces to optimize")

    swap_pairs = list(combinations(range(S), 2))
    neighborhood_size = len(swap_pairs)

    # Resolve tenure: fixed int > coefficient k > neighborhood fraction > default 0.05 * C(S,2)
    if tabu_tenure is not None:
        tenure = int(tabu_tenure)
    elif tabu_tenure_coefficient is not None:
        k = float(tabu_tenure_coefficient)
        tenure = max(3, int(_math.ceil(k * (S ** 0.5))))
    elif tabu_tenure_neighborhood_fraction is not None:
        alpha = float(tabu_tenure_neighborhood_fraction)
        tenure = max(3, int(_math.ceil(alpha * neighborhood_size)))
    else:
        # Default: scale with neighborhood size (plan: 0.05 * C(S,2))
        tenure = max(3, int(_math.ceil(0.05 * neighborhood_size)))
    tabu_tenure = tenure

    eval_kw = dict(
        normalizer_sigma=normalizer_sigma,
        use_smooth_objectives=use_smooth_objectives,
        smooth_p=smooth_p,
        smooth_k=smooth_k,
        use_local_variance_lisa=use_local_variance_lisa,
    )
    current_score = evaluate_map_multiobjective(ti4_map, evaluator, weights, fast_state, **eval_kw)
    best_score = current_score
    best_lex = best_score.lex_key()
    best_etb = 0
    history: List[Tuple[int, MultiObjectiveScore]] = [(0, current_score)]

    # (min_idx, max_idx) → iteration number when the tabu expires
    tabu_dict: Dict[Tuple[int, int], int] = {}
    # (s_pos, system_id) → expiry iteration (when use_attribute_tabu)
    tabu_attrs: Dict[Tuple[int, int], int] = {}

    total_evals = 0
    iteration = 0
    iterations_since_improvement = 0

    if verbose:
        print(
            f"TS: S={S}, neighborhood={neighborhood_size}, "
            f"tenure={tabu_tenure}, budget={max_evaluations}"
        )
        print(f"Initial: {current_score}")

    while total_evals + neighborhood_size <= max_evaluations:
        iteration += 1

        best_move: Optional[Tuple[int, int]] = None
        best_move_score: Optional[MultiObjectiveScore] = None
        best_move_lex = (float('inf'), float('inf'))

        best_tabu_move: Optional[Tuple[int, int]] = None
        best_tabu_score: Optional[MultiObjectiveScore] = None
        best_tabu_lex = (float('inf'), float('inf'))

        for s_i, s_j in swap_pairs:
            fast_state.swap(s_i, s_j)
            candidate = evaluate_map_multiobjective(
                ti4_map, evaluator, weights, fast_state, **eval_kw
            )
            c_lex = candidate.lex_key()
            total_evals += 1

            is_tabu_pair = (s_i, s_j) in tabu_dict and tabu_dict[(s_i, s_j)] > iteration
            is_tabu_attr = False
            if use_attribute_tabu:
                # Forbid if placing system at s_j onto s_i (or system at s_i onto s_j) is tabu
                id_at_j = swappable_spaces[s_j].system.id
                id_at_i = swappable_spaces[s_i].system.id
                is_tabu_attr = (
                    ((s_i, id_at_j) in tabu_attrs and tabu_attrs[(s_i, id_at_j)] > iteration)
                    or ((s_j, id_at_i) in tabu_attrs and tabu_attrs[(s_j, id_at_i)] > iteration)
                )
            is_tabu = is_tabu_pair or is_tabu_attr

            if is_tabu:
                if c_lex < best_tabu_lex:
                    best_tabu_lex = c_lex
                    best_tabu_move = (s_i, s_j)
                    best_tabu_score = candidate
            else:
                if c_lex < best_move_lex:
                    best_move_lex = c_lex
                    best_move = (s_i, s_j)
                    best_move_score = candidate

            fast_state.swap(s_i, s_j)  # revert trial

        # Aspiration: allow tabu move if it beats global best
        chosen_move = best_move
        chosen_score = best_move_score

        if best_tabu_move is not None and best_tabu_lex < best_lex:
            chosen_move = best_tabu_move
            chosen_score = best_tabu_score

        if chosen_move is None:
            # All non-tabu moves exhausted and no aspiration — fallback to best tabu
            if best_tabu_move is not None:
                chosen_move = best_tabu_move
                chosen_score = best_tabu_score
            else:
                break

        # Apply chosen move to both fast_state and ti4_map
        s_i, s_j = chosen_move
        fast_state.swap(s_i, s_j)
        swappable_spaces[s_i].system, swappable_spaces[s_j].system = (
            swappable_spaces[s_j].system, swappable_spaces[s_i].system
        )
        current_score = chosen_score

        tabu_dict[(s_i, s_j)] = iteration + tabu_tenure
        if use_attribute_tabu:
            tabu_attrs[(s_i, swappable_spaces[s_i].system.id)] = iteration + tabu_tenure
            tabu_attrs[(s_j, swappable_spaces[s_j].system.id)] = iteration + tabu_tenure

        if current_score.lex_key() < best_lex:
            best_score = current_score
            best_lex = best_score.lex_key()
            best_etb = total_evals
            iterations_since_improvement = 0
            if verbose:
                print(
                    f"  Iter {iteration} (evals={total_evals}): "
                    f"NEW BEST {best_score}"
                )
        else:
            iterations_since_improvement += 1

        history.append((total_evals, current_score))

        # Stagnation diversification: random swaps to escape plateaus
        if stagnation_threshold is not None and iterations_since_improvement >= stagnation_threshold:
            n_swaps = random.randint(1, 3)
            for _ in range(n_swaps):
                a, b = random.sample(range(S), 2)
                fast_state.swap(a, b)
                swappable_spaces[a].system, swappable_spaces[b].system = (
                    swappable_spaces[b].system, swappable_spaces[a].system
                )
            if total_evals < max_evaluations:
                current_score = evaluate_map_multiobjective(
                    ti4_map, evaluator, weights, fast_state, **eval_kw
                )
                total_evals += 1
                history.append((total_evals, current_score))
                if current_score.lex_key() < best_lex:
                    best_score = current_score
                    best_lex = best_score.lex_key()
                    best_etb = total_evals
                    iterations_since_improvement = 0
            iterations_since_improvement = 0  # reset after diversification

    # Use remaining budget for a partial neighbourhood scan
    remaining = max_evaluations - total_evals
    if remaining > 0 and len(swap_pairs) > 0:
        iteration += 1
        partial_pairs = swap_pairs.copy()
        random.shuffle(partial_pairs)
        partial_pairs = partial_pairs[:remaining]

        best_move = None
        best_move_score = None
        best_move_lex = (float('inf'), float('inf'))
        best_tabu_move = None
        best_tabu_score = None
        best_tabu_lex = (float('inf'), float('inf'))

        for s_i, s_j in partial_pairs:
            fast_state.swap(s_i, s_j)
            candidate = evaluate_map_multiobjective(
                ti4_map, evaluator, weights, fast_state, **eval_kw
            )
            c_lex = candidate.lex_key()
            total_evals += 1

            is_tabu_pair = (s_i, s_j) in tabu_dict and tabu_dict[(s_i, s_j)] > iteration
            is_tabu_attr = False
            if use_attribute_tabu:
                id_at_j = swappable_spaces[s_j].system.id
                id_at_i = swappable_spaces[s_i].system.id
                is_tabu_attr = (
                    ((s_i, id_at_j) in tabu_attrs and tabu_attrs[(s_i, id_at_j)] > iteration)
                    or ((s_j, id_at_i) in tabu_attrs and tabu_attrs[(s_j, id_at_i)] > iteration)
                )
            is_tabu = is_tabu_pair or is_tabu_attr

            if is_tabu:
                if c_lex < best_tabu_lex:
                    best_tabu_lex = c_lex
                    best_tabu_move = (s_i, s_j)
                    best_tabu_score = candidate
            else:
                if c_lex < best_move_lex:
                    best_move_lex = c_lex
                    best_move = (s_i, s_j)
                    best_move_score = candidate

            fast_state.swap(s_i, s_j)

        chosen_move = best_move
        chosen_score = best_move_score
        if best_tabu_move is not None and best_tabu_lex < best_lex:
            chosen_move = best_tabu_move
            chosen_score = best_tabu_score
        if chosen_move is None and best_tabu_move is not None:
            chosen_move = best_tabu_move
            chosen_score = best_tabu_score

        if chosen_move is not None:
            s_i, s_j = chosen_move
            fast_state.swap(s_i, s_j)
            swappable_spaces[s_i].system, swappable_spaces[s_j].system = (
                swappable_spaces[s_j].system, swappable_spaces[s_i].system
            )
            current_score = chosen_score
            tabu_dict[(s_i, s_j)] = iteration + tabu_tenure
            if use_attribute_tabu:
                tabu_attrs[(s_i, swappable_spaces[s_i].system.id)] = iteration + tabu_tenure
                tabu_attrs[(s_j, swappable_spaces[s_j].system.id)] = iteration + tabu_tenure
            if current_score.lex_key() < best_lex:
                best_score = current_score
                best_lex = best_score.lex_key()
                best_etb = total_evals
            history.append((total_evals, current_score))

    if verbose:
        print(f"TS complete: {iteration} iterations, {total_evals} evaluations")
        print(f"Final best: {best_score}")

    history.append((total_evals, best_score))
    return best_score, history, best_etb, iteration
