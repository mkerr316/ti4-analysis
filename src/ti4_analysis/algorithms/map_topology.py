"""
Static pre-computed topology for a fixed TI4 map template.

The hex grid topology (which positions are adjacent, which are blocked by
Supernovas, which are connected via wormholes) is immutable during an
optimization run because anomaly and wormhole tiles are excluded from
swapping. This allows the distance-weighted evaluation to be baked into
a static (H x S) weight matrix computed once at optimizer initialization.

After construction, fitness evaluation reduces to:
    home_values = weight_matrix @ system_value   # single C-level matmul
"""

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import scipy.sparse

from ..data.map_structures import Evaluator, MapSpaceType
from .hex_grid import hex_distance

if TYPE_CHECKING:
    from .balance_engine import TI4Map, can_swap_system


@dataclass(frozen=True, eq=False)
class MapTopology:
    """
    Immutable topology for a specific map template and evaluator.

    Build once via MapTopology.from_ti4_map(); reuse for all swap iterations.

    Home value decomposition:
        home_values = static_home_values + dynamic_weight_matrix @ system_value

    where:
        static_home_values  — contribution from non-swappable systems (Mecatol,
                              wormhole tiles, anomaly tiles, etc.). Constant for
                              the entire optimization run.
        dynamic_weight_matrix — per-swappable-position weights, shape (H, S).
        system_value        — current evaluator score at each swappable position,
                              maintained by FastMapState and updated O(1) per swap.

    Spatial metric decomposition (Moran's I):
        z = spatial_static_values + spatial_projection @ system_value

    where z is the full N_sys-length resource values vector over all system spaces.
    spatial_W is a pre-computed row-standardized sparse adjacency matrix used for
    vectorized Moran's I computation in FastMapState.

    Multi-Jain decomposition (per-dimension JFI):
        home_resources = static_home_resources + dynamic_weight_matrix @ system_resources
        home_influence = static_home_influence + dynamic_weight_matrix @ system_influence

    where system_resources/influence hold raw planet totals (no evaluator
    multipliers) and the same dynamic_weight_matrix encodes distance weights.

    Attributes:
        home_indices: Indices into TI4Map.spaces for home positions, shape (H,).
        swappable_indices: Indices into TI4Map.spaces for swappable system
            positions, shape (S,). Ordering matches FastMapState.system_value.
        static_home_values: Pre-summed contribution of non-swappable systems,
            shape (H,). Added back in FastMapState.home_values().
        static_home_resources: Pre-summed raw resource contribution from
            non-swappable systems, shape (H,). Used for per-dimension JFI.
        static_home_influence: Pre-summed raw influence contribution from
            non-swappable systems, shape (H,). Used for per-dimension JFI.
        dynamic_weight_matrix: Pre-computed fitness weights for swappable
            positions, shape (H, S).
            dynamic_weight_matrix[h, s] = evaluator.get_distance_multiplier(
                modded_routing_distance(home_h, swappable_s))
            0.0 where Supernova-blocked or geometric distance >= 5.
        spatial_indices: Indices into TI4Map.spaces for all system spaces with
            non-None systems, shape (N_sys,). Ordering matches rows of spatial_W.
        spatial_static_values: Resource value for non-swappable system positions,
            0.0 for swappable positions, shape (N_sys,) float32.
        spatial_projection: Sparse (N_sys, S) matrix mapping system_value to the
            swappable portion of the full spatial values vector.
        spatial_W: Row-standardized sparse binary adjacency matrix (N_sys, N_sys).
            Zero-degree (island) hexes are purged so n and variance match the lag.
        spatial_W_swappable: Row-standardized adjacency over swappable positions only
            (S_conn, S_conn). Decision variables only; no static outliers in variance.
        swappable_connected_s_pos: Indices into swappable_indices (0..S-1) that have
            at least one swappable neighbor, shape (S_conn,). Used for morans_i_swappable.
    """

    home_indices: np.ndarray          # shape (H,) int32
    swappable_indices: np.ndarray     # shape (S,) int32
    static_home_values: np.ndarray    # shape (H,) float32
    static_home_resources: np.ndarray # shape (H,) float32
    static_home_influence: np.ndarray # shape (H,) float32
    dynamic_weight_matrix: np.ndarray # shape (H, S) float32
    spatial_indices: np.ndarray       # shape (N_sys,) int32
    spatial_static_values: np.ndarray # shape (N_sys,) float32
    spatial_projection: object = field(compare=False)  # scipy.sparse.csr_matrix (N_sys, S)
    spatial_W: object = field(compare=False)           # scipy.sparse.csr_matrix (N_sys, N_sys)
    spatial_W_swappable: object = field(compare=False)  # scipy.sparse.csr_matrix (S_conn, S_conn)
    swappable_connected_s_pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))  # (S_conn,)
    degree_swappable: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))  # (S_conn,) degree k_i per node for LSAP sqrt(k_i) correction

    @classmethod
    def from_ti4_map(cls, ti4_map: 'TI4Map', evaluator: Evaluator) -> 'MapTopology':
        """
        Build topology by running get_shortest_modded_distance once per
        (home, system) pair. O(H * N_systems * BFS) — runs once at optimizer init.

        Non-swappable systems (Mecatol, wormhole tiles, anomaly tiles, empty
        tiles with planets) contribute a fixed amount to each home value and are
        baked into static_home_values. Only swappable system positions are
        included in dynamic_weight_matrix.

        Args:
            ti4_map: Map whose topology to pre-compute.
            evaluator: Evaluator controlling anomaly traversal costs.

        Returns:
            MapTopology instance ready for use with FastMapState.
        """
        from .balance_engine import can_swap_system

        spaces = ti4_map.spaces

        home_indices = np.array(
            [i for i, s in enumerate(spaces) if s.space_type == MapSpaceType.HOME],
            dtype=np.int32,
        )
        swappable_indices = np.array(
            [i for i, s in enumerate(spaces) if can_swap_system(s)],
            dtype=np.int32,
        )
        swappable_set = set(swappable_indices.tolist())

        H = len(home_indices)
        S = len(swappable_indices)
        static_home_values = np.zeros(H, dtype=np.float32)
        static_home_resources = np.zeros(H, dtype=np.float32)
        static_home_influence = np.zeros(H, dtype=np.float32)
        dynamic_weight_matrix = np.zeros((H, S), dtype=np.float32)

        # s_pos lookup: space index → column in dynamic_weight_matrix
        s_pos_of = {int(idx): pos for pos, idx in enumerate(swappable_indices)}

        for h_pos, h_idx in enumerate(home_indices):
            home_space = spaces[h_idx]

            for space_idx, space in enumerate(spaces):
                if space.space_type != MapSpaceType.SYSTEM or space.system is None:
                    continue

                # Geometric range gate — matches get_home_value's cutoff
                if hex_distance(home_space.coord, space.coord) >= 5:
                    continue

                system_value = space.system.evaluate(evaluator)
                if system_value <= 0:
                    continue

                # Topological routing distance (None if blocked by Supernova)
                modded_dist = ti4_map.get_shortest_modded_distance(
                    home_space, space, evaluator
                )
                if modded_dist is None:
                    continue

                weight = evaluator.get_distance_multiplier(modded_dist)

                if space_idx in swappable_set:
                    # Dynamic: tile value changes during optimization
                    dynamic_weight_matrix[h_pos, s_pos_of[space_idx]] = weight
                else:
                    # Static: tile never moves; bake value × weight into constant
                    static_home_values[h_pos] += weight * system_value
                    sys_res = sum(p.resources for p in space.system.planets)
                    sys_inf = sum(p.influence for p in space.system.planets)
                    static_home_resources[h_pos] += weight * sys_res
                    static_home_influence[h_pos] += weight * sys_inf

        # --- Spatial metrics pre-computation ---
        # Collect all system spaces that have a system object (N_sys positions).
        spatial_indices = np.array(
            [
                i for i, s in enumerate(spaces)
                if s.space_type == MapSpaceType.SYSTEM and s.system is not None
            ],
            dtype=np.int32,
        )
        N_sys = len(spatial_indices)

        # Build a lookup from space index → row in the spatial arrays.
        spatial_row_of = {int(idx): row for row, idx in enumerate(spatial_indices)}

        # static values: non-swappable tiles contribute their eval score; swappable = 0.
        spatial_static_values = np.array(
            [
                0.0 if int(spatial_indices[row]) in swappable_set
                else spaces[int(spatial_indices[row])].system.evaluate(evaluator)
                for row in range(N_sys)
            ],
            dtype=np.float32,
        )

        # Sparse projection matrix (N_sys, S): 1.0 at [row, s_col] for swappable positions.
        proj_rows, proj_cols, proj_data = [], [], []
        for s_col, space_idx in enumerate(swappable_indices):
            if int(space_idx) in spatial_row_of:
                proj_rows.append(spatial_row_of[int(space_idx)])
                proj_cols.append(s_col)
                proj_data.append(1.0)
        spatial_projection = scipy.sparse.csr_matrix(
            (proj_data, (proj_rows, proj_cols)),
            shape=(N_sys, len(swappable_indices)),
            dtype=np.float32,
        )

        # Sparse binary adjacency matrix (N_sys, N_sys) on navigable topology only.
        # Skip edges from or to impassable tiles (e.g. Supernova) so Moran's I / LSAP
        # use strategic proximity, not geometric adjacency.
        adj_rows, adj_cols = [], []
        for row_k, space_idx in enumerate(spatial_indices):
            space = spaces[int(space_idx)]
            if space.system is not None and space.system.get_distance_modifier(evaluator) is None:
                continue
            neighbors = ti4_map.get_adjacent_spaces_including_wormholes(space)
            for nb in neighbors:
                if nb.system is not None and nb.system.get_distance_modifier(evaluator) is None:
                    continue
                nb_idx = next(
                    (i for i, s in enumerate(spaces) if s is nb), None
                )
                if nb_idx is not None and nb_idx in spatial_row_of:
                    adj_rows.append(row_k)
                    adj_cols.append(spatial_row_of[nb_idx])
        W_raw = scipy.sparse.csr_matrix(
            ([1.0] * len(adj_rows), (adj_rows, adj_cols)),
            shape=(N_sys, N_sys),
            dtype=np.float32,
        )

        # Purge zero-degree nodes (island hexes). They contribute to variance
        # but not to spatial lag, corrupting LSAP. Keep only connected nodes.
        row_sums = np.array(W_raw.sum(axis=1)).flatten()
        keep = row_sums > 0
        keep_inds = np.where(keep)[0]
        spatial_indices = spatial_indices[keep]
        spatial_static_values = spatial_static_values[keep]
        spatial_projection = spatial_projection[keep_inds, :]
        W_kept = W_raw[keep_inds, :][:, keep_inds]
        N_sys = len(spatial_indices)
        row_sums_kept = np.array(W_kept.sum(axis=1)).flatten()
        # After purge, no zero rows remain.
        spatial_W = scipy.sparse.diags(1.0 / row_sums_kept) @ W_kept
        spatial_W = scipy.sparse.csr_matrix(spatial_W, dtype=np.float32)

        # Swappable-only adjacency (S x S) for Moran's I / LSAP over decision variables.
        # Same navigability rules; then purge zero-degree so variance = lag domain.
        S = len(swappable_indices)
        sw_pos_of = {int(swappable_indices[p]): p for p in range(S)}
        adj_sw_r, adj_sw_c = [], []
        for s_pos in range(S):
            space_idx = int(swappable_indices[s_pos])
            space = spaces[space_idx]
            if space.system is not None and space.system.get_distance_modifier(evaluator) is None:
                continue
            for nb in ti4_map.get_adjacent_spaces_including_wormholes(space):
                if nb.system is not None and nb.system.get_distance_modifier(evaluator) is None:
                    continue
                nb_idx = next((i for i, s in enumerate(spaces) if s is nb), None)
                if nb_idx is not None and nb_idx in sw_pos_of:
                    adj_sw_c.append(sw_pos_of[nb_idx])
                    adj_sw_r.append(s_pos)
        W_sw_raw = scipy.sparse.csr_matrix(
            ([1.0] * len(adj_sw_r), (adj_sw_r, adj_sw_c)),
            shape=(S, S),
            dtype=np.float32,
        )
        row_sums_sw = np.array(W_sw_raw.sum(axis=1)).flatten()
        keep_sw = row_sums_sw > 0
        keep_sw_inds = np.where(keep_sw)[0].astype(np.int32)
        W_sw_kept = W_sw_raw[keep_sw_inds, :][:, keep_sw_inds]
        row_sums_sw_kept = np.array(W_sw_kept.sum(axis=1)).flatten()
        spatial_W_swappable = scipy.sparse.diags(1.0 / row_sums_sw_kept) @ W_sw_kept
        spatial_W_swappable = scipy.sparse.csr_matrix(spatial_W_swappable, dtype=np.float32)

        # Degree (row sum of binary W) per swappable-connected node for LSAP local-variance correction.
        degree_swappable = row_sums_sw_kept.astype(np.float32)

        return cls(
            home_indices=home_indices,
            swappable_indices=swappable_indices,
            static_home_values=static_home_values,
            static_home_resources=static_home_resources,
            static_home_influence=static_home_influence,
            dynamic_weight_matrix=dynamic_weight_matrix,
            spatial_indices=spatial_indices,
            spatial_static_values=spatial_static_values,
            spatial_projection=spatial_projection,
            spatial_W=spatial_W,
            spatial_W_swappable=spatial_W_swappable,
            swappable_connected_s_pos=keep_sw_inds,
            degree_swappable=degree_swappable,
        )
