"""
Balance Engine for TI4 Map Generator.

Implements the ti4-map-lab iterative optimization algorithm for
minimizing player advantage gaps through strategic system swapping.

Algorithm:
    1. Calculate "home values" for each player position
    2. Compute balance gap (max - min home values)
    3. Randomly swap planet-containing systems
    4. Accept swaps that reduce the gap (greedy hill climbing)
    5. Repeat for N iterations

Reference: src/balance/balance-engine.js
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import random
import copy

from ..algorithms.hex_grid import HexCoord, hex_distance, breadth_first_search, get_adjacent_coordinates
from ..data.map_structures import (
    MapSpace, MapSpaceType, System, Evaluator, Anomaly, Wormhole
)
from .map_topology import MapTopology
from .fast_map_state import FastMapState


@dataclass
class HomeValue:
    """Represents the computed value for a home system position."""
    space: MapSpace
    value: float


class TI4Map:
    """
    Represents a complete TI4 map with systems and home positions.

    This is the main class for map balancing operations.
    """

    def __init__(self, spaces: List[MapSpace]):
        """
        Initialize map with spaces.

        Args:
            spaces: List of MapSpace objects
        """
        self.spaces = spaces
        self._coord_to_space: Dict[HexCoord, MapSpace] = {
            space.coord: space for space in spaces
        }

    def get_space(self, coord: HexCoord) -> Optional[MapSpace]:
        """Get space at given coordinate."""
        return self._coord_to_space.get(coord)

    def get_home_spaces(self) -> List[MapSpace]:
        """Get all home system spaces."""
        return [s for s in self.spaces if s.space_type == MapSpaceType.HOME]

    def get_system_spaces(self) -> List[MapSpace]:
        """Get all system spaces (excluding open/closed)."""
        return [s for s in self.spaces
                if s.space_type == MapSpaceType.SYSTEM and s.system is not None]

    def get_adjacent_spaces(self, space: MapSpace) -> List[MapSpace]:
        """
        Get all adjacent spaces (excluding wormhole connections).

        Args:
            space: Center space

        Returns:
            List of adjacent MapSpace objects
        """
        adjacent_coords = get_adjacent_coordinates(space.coord)
        adjacent_spaces = []

        for coord in adjacent_coords:
            adj_space = self.get_space(coord)
            if adj_space is not None:
                adjacent_spaces.append(adj_space)

        return adjacent_spaces

    def get_adjacent_spaces_including_wormholes(self, space: MapSpace) -> List[MapSpace]:
        """
        Get all adjacent spaces including wormhole connections.

        Wormholes of the same type are treated as adjacent.

        Args:
            space: Center space

        Returns:
            List of adjacent MapSpace objects (including wormhole connections)
        """
        adjacent_spaces = self.get_adjacent_spaces(space)

        # Add wormhole connections
        if space.space_type == MapSpaceType.SYSTEM and space.system and space.system.wormhole:
            wormhole_type = space.system.wormhole
            for other_space in self.spaces:
                if (other_space.space_type == MapSpaceType.SYSTEM and
                    other_space.system and
                    other_space.system.wormhole == wormhole_type and
                    other_space.system.id != space.system.id):
                    if other_space not in adjacent_spaces:
                        adjacent_spaces.append(other_space)

        return adjacent_spaces

    def get_shortest_modded_distance(
        self,
        start: MapSpace,
        dest: MapSpace,
        evaluator: Evaluator,
        max_path_length: int = 5
    ) -> Optional[float]:
        """
        Calculate shortest modified distance using pathfinding.

        This accounts for:
        - Wormhole shortcuts
        - Anomaly penalties
        - System-specific modifiers

        Args:
            start: Starting space
            dest: Destination space
            evaluator: Evaluator with distance modifiers
            max_path_length: Maximum path length to search

        Returns:
            Modified distance, or None if no valid path
        """
        finished_paths: List[List[int]] = []
        active_paths: List[List[int]] = []

        # Initialize with first step from start
        first_steps = self.get_adjacent_spaces_including_wormholes(start)

        for one_step in first_steps:
            if one_step.system is None:
                continue

            dist_mod = one_step.system.get_distance_modifier(evaluator)
            if dist_mod is None:  # System blocks path
                continue

            new_path = [one_step.system.id]

            if one_step.system.id == dest.system.id:
                finished_paths.append(new_path)
            else:
                active_paths.append(new_path)

        # Extend paths iteratively
        while active_paths:
            new_active_paths = []

            for path in active_paths:
                results = self._extend_path(start, dest, path, evaluator, max_path_length)
                finished_paths.extend(results['finished'])
                new_active_paths.extend(results['ongoing'])

            active_paths = new_active_paths

        # Find shortest path
        if not finished_paths:
            return None

        shortest = None
        for path in finished_paths:
            path_length = self._calculate_modded_distance_from_path(path, evaluator, start)
            if path_length is not None and (shortest is None or path_length < shortest):
                shortest = path_length

        return shortest

    def _extend_path(
        self,
        start: MapSpace,
        dest: MapSpace,
        path: List[int],
        evaluator: Evaluator,
        max_length: int
    ) -> Dict[str, List[List[int]]]:
        """
        Extend a path by one step.

        Returns:
            Dict with 'finished' and 'ongoing' path lists
        """
        completed_paths = []
        ongoing_paths = []

        last_space = self._get_space_by_system_id(path[-1])
        if last_space is None:
            return {'finished': completed_paths, 'ongoing': ongoing_paths}

        next_steps = self.get_adjacent_spaces_including_wormholes(last_space)

        for one_step in next_steps:
            if one_step.system is None:
                continue

            if one_step.system.id == dest.system.id:
                path_copy = path + [one_step.system.id]
                completed_paths.append(path_copy)
            elif (one_step.system.id not in path and
                  one_step.system.get_distance_modifier(evaluator) is not None and
                  len(path) < max_length):
                path_copy = path + [one_step.system.id]
                ongoing_paths.append(path_copy)

        return {'finished': completed_paths, 'ongoing': ongoing_paths}

    def _calculate_modded_distance_from_path(
        self,
        path: List[int],
        evaluator: Evaluator,
        start: MapSpace
    ) -> Optional[float]:
        """Calculate total modified distance for a path."""
        modded_dist = 0.0

        for i, system_id in enumerate(path):
            space = self._get_space_by_system_id(system_id)
            if space is None or space.system is None:
                return None

            # Check if accessed through wormhole
            through_wh = False
            if i > 0:
                # If not adjacent normally, must be through wormhole
                if space not in self.get_adjacent_spaces(start):
                    through_wh = True

            dist_mod = space.system.get_distance_modifier(evaluator, through_wh)
            if dist_mod is None:
                return None

            # Gravity Rift: use Expected Effort modifier M = 0.3*d_rift - 0.4
            # (d_rift = hops from start to this system = i+1). See docs/limitations/anomalies.md.
            if space.system.anomalies and Anomaly.GRAVITY_RIFT in space.system.anomalies:
                static_rift = evaluator.DISTANCE_MOD_GRAVITY_RIFT
                if static_rift is not False and static_rift is not None:
                    static_rift = float(static_rift)
                else:
                    static_rift = 0.0
                dist_mod = dist_mod - static_rift + (0.3 * (i + 1) - 0.4)

            modded_dist += dist_mod

        return modded_dist

    def _get_space_by_system_id(self, system_id: int) -> Optional[MapSpace]:
        """Find space containing system with given ID."""
        for space in self.spaces:
            if (space.space_type == MapSpaceType.SYSTEM and
                space.system and space.system.id == system_id):
                return space
        return None

    def get_home_value(self, home_space: MapSpace, evaluator: Evaluator) -> float:
        """
        Calculate total value accessible from a home system.

        This is the core metric for balance: sum of system values
        weighted by distance-adjusted accessibility.

        Formula:
            HomeValue = Σ (SystemValue × DistanceMultiplier(distance))
            for all systems within range

        Args:
            home_space: Home system space
            evaluator: Evaluator with parameters

        Returns:
            Total home value
        """
        home_total = 0.0

        for space in self.spaces:
            if space.space_type != MapSpaceType.SYSTEM or space.system is None:
                continue

            # Only consider systems within reasonable distance
            cube_distance = hex_distance(home_space.coord, space.coord)
            if cube_distance >= 5:
                continue

            # Evaluate system value
            system_value = space.system.evaluate(evaluator)
            if system_value <= 0:
                continue

            # Find shortest modified path distance
            shortest_distance = self.get_shortest_modded_distance(
                home_space, space, evaluator
            )

            if shortest_distance is not None:
                distance_mult = evaluator.get_distance_multiplier(shortest_distance)
                home_total += distance_mult * system_value

        return home_total

    def copy(self) -> 'TI4Map':
        """Create a deep copy of this map."""
        new_spaces = []
        for space in self.spaces:
            new_space = MapSpace(
                coord=space.coord,
                space_type=space.space_type,
                system=space.system  # Systems are shared, not copied
            )
            new_spaces.append(new_space)
        return TI4Map(new_spaces)


def get_home_values(ti4_map: TI4Map, evaluator: Evaluator) -> List[HomeValue]:
    """
    Calculate home values for all home systems.

    Args:
        ti4_map: TI4 map object
        evaluator: Evaluation parameters

    Returns:
        List of HomeValue objects
    """
    home_values = []

    for space in ti4_map.get_home_spaces():
        value = ti4_map.get_home_value(space, evaluator)
        home_values.append(HomeValue(space=space, value=value))

    return home_values


def get_balance_gap(home_values: List[HomeValue]) -> float:
    """
    Calculate balance gap (max - min home values).

    A smaller gap means more balanced player positions.

    Args:
        home_values: List of home values

    Returns:
        Balance gap value
    """
    if not home_values:
        return 0.0

    values = [hv.value for hv in home_values]
    return max(values) - min(values)


def can_swap_system(space: MapSpace) -> bool:
    """
    Check if a system can be swapped during balancing.

    We avoid swapping:
    - Wormhole systems (preserves connectivity)
    - Anomaly systems (preserves strategic features)
    - Empty systems (no value to swap)
    - Mecatol Rex (center of the map)

    Args:
        space: MapSpace to check

    Returns:
        True if system can be swapped
    """
    if space.space_type != MapSpaceType.SYSTEM:
        return False

    if space.system is None:
        return False

    # Don't swap wormhole systems
    if space.system.wormhole is not None:
        return False

    # Don't swap anomaly systems
    if space.system.anomalies is not None:
        return False

    # Don't swap empty systems
    if len(space.system.planets) == 0:
        return False

    # Don't swap Mecatol Rex
    if space.system.is_mecatol_rex():
        return False

    return True


def improve_balance(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    iterations: int = 100,
    random_seed: Optional[int] = None
) -> Tuple[float, List[Tuple[int, float]]]:
    """
    Improve map balance through iterative system swapping.

    Algorithm:
        1. Identify swappable systems (excluding wormholes, anomalies, etc.)
        2. For N iterations:
            a. Randomly select two swappable systems
            b. Swap them
            c. Recalculate balance gap
            d. If improved, keep swap; otherwise revert

    This is a greedy hill-climbing algorithm that may find local minima.

    Fitness evaluation uses a pre-computed static weight matrix (MapTopology)
    and a vectorized numpy state (FastMapState). The BFS that previously ran
    on every iteration runs exactly once at initialization; each subsequent
    evaluation is a single C-level matrix-vector product.

    Args:
        ti4_map: TI4 map object (will be modified in-place)
        evaluator: Evaluation parameters
        iterations: Number of swap attempts
        random_seed: Optional seed for reproducibility

    Returns:
        Tuple of (final_balance_gap, history)
        where history is [(iteration, balance_gap), ...]
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Build static topology once; initialize vectorized state.
    # swappable_spaces[s] corresponds to fast_state.system_value[s].
    topology = MapTopology.from_ti4_map(ti4_map, evaluator)
    fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)
    swappable_spaces = [ti4_map.spaces[i] for i in topology.swappable_indices]

    if len(swappable_spaces) < 2:
        print("Warning: Not enough swappable systems for balancing")
        balance_gap = get_balance_gap(get_home_values(ti4_map, evaluator))
        return balance_gap, [(0, balance_gap)]

    balance_gap = fast_state.balance_gap()
    history = [(0, balance_gap)]
    swaps_accepted = 0

    for i in range(1, iterations + 1):
        # Pick two random swappable-position indices
        s1, s2 = random.sample(range(len(swappable_spaces)), 2)

        # Trial swap in fast state (O(1))
        fast_state.swap(s1, s2)
        new_balance_gap = fast_state.balance_gap()

        if new_balance_gap < balance_gap:
            # Accept: apply the same swap to ti4_map (keeps it in sync)
            swappable_spaces[s1].system, swappable_spaces[s2].system = (
                swappable_spaces[s2].system,
                swappable_spaces[s1].system,
            )
            balance_gap = new_balance_gap
            swaps_accepted += 1
        else:
            # Revert fast state (swap is its own inverse)
            fast_state.swap(s1, s2)

        history.append((i, balance_gap))

    print(f"Balance optimization complete: {swaps_accepted}/{iterations} swaps accepted")
    print(f"Final balance gap: {balance_gap:.2f}")

    return balance_gap, history


def analyze_balance(
    ti4_map: TI4Map,
    evaluator: Evaluator
) -> Dict[str, any]:
    """
    Analyze current map balance without making changes.

    Args:
        ti4_map: TI4 map object
        evaluator: Evaluation parameters

    Returns:
        Dictionary with balance statistics:
        - home_values: List of (position, value) tuples
        - balance_gap: Max - min values
        - mean: Average home value
        - std: Standard deviation
        - fairness_index: Jain's fairness index
    """
    import numpy as np

    home_values = get_home_values(ti4_map, evaluator)
    values = [hv.value for hv in home_values]

    if not values:
        return {
            'home_values': [],
            'balance_gap': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'fairness_index': 1.0
        }

    values_array = np.array(values)

    # Jain's Fairness Index: (Σx)² / (n × Σx²)
    # Range: [1/n, 1], where 1 is perfectly fair
    fairness_index = (values_array.sum() ** 2) / (len(values) * (values_array ** 2).sum())

    return {
        'home_values': [(i, v) for i, v in enumerate(values)],
        'balance_gap': max(values) - min(values),
        'mean': values_array.mean(),
        'std': values_array.std(),
        'min': values_array.min(),
        'max': values_array.max(),
        'fairness_index': fairness_index
    }
