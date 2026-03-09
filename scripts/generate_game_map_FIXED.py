#!/usr/bin/env python3
"""
TI4 MAP GENERATOR - FIXED VERSION
==================================

KEY FIXES:
1. Added --selection flag: min_slv (default), min_gap, knee, balanced
2. Removed broken retry logic - generates once and exports
3. Proper Pareto selection prioritizing SLV

Usage:
    python generate_game_map_FIXED.py --players 6
    python generate_game_map_FIXED.py --players 6 --selection knee
    python generate_game_map_FIXED.py --players 6 --selection balanced
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json


from ti4_analysis.algorithms.map_generator import generate_random_map
from ti4_analysis.data.ti4proj_exporter import (
    export_to_ti4proj,
    validate_ti4proj,
    hexcoord_to_axial
)
from ti4_analysis.algorithms.balance_engine import (
    TI4Map,
    create_joebrew_evaluator,
    get_home_values,
    get_balance_gap
)
from ti4_analysis.algorithms.nsga2_optimizer import optimize_map_nsga2
from ti4_analysis.spatial_stats.spatial_metrics import (
    jains_fairness_index,
    resource_clustering_coefficient,
    compute_slice_level_variance
)
from ti4_analysis.data.map_structures import MapSpaceType, System
from typing import Dict, List
import numpy as np
import random


def select_from_pareto_front(
    pareto_objectives: np.ndarray,
    pareto_maps: List[TI4Map],
    method: str = "min_slv"
) -> tuple[TI4Map, int, Dict]:
    """
    Select solution from Pareto front.
    
    Args:
        pareto_objectives: (n, 3) array with [gap, moran_i, slv]
        pareto_maps: List of TI4Map objects
        method: "min_slv", "min_gap", "knee", or "balanced"
    
    Returns:
        (selected_map, selected_index, metrics_dict)
    """
    
    if method == "min_slv":
        # Prioritize local fairness (what you want!)
        idx = np.argmin(pareto_objectives[:, 2])
        
    elif method == "min_gap":
        # Prioritize global balance  
        idx = np.argmin(pareto_objectives[:, 0])
        
    elif method == "knee":
        # Knee point: closest to ideal (0, 0, 0)
        normalized = (pareto_objectives - pareto_objectives.min(axis=0)) / (
            pareto_objectives.max(axis=0) - pareto_objectives.min(axis=0) + 1e-8
        )
        distances = np.sqrt(np.sum(normalized**2, axis=1))
        idx = np.argmin(distances)
        
    elif method == "balanced":
        # Tchebyshev with equal weights
        normalized = (pareto_objectives - pareto_objectives.min(axis=0)) / (
            pareto_objectives.max(axis=0) - pareto_objectives.min(axis=0) + 1e-8
        )
        weights = np.array([0.33, 0.33, 0.34])
        rho = 0.05
        scores = np.max(weights * normalized, axis=1) + rho * np.sum(normalized, axis=1)
        idx = np.argmin(scores)
        
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    return pareto_maps[idx], idx, {
        'gap': float(pareto_objectives[idx, 0]),
        'moran': float(pareto_objectives[idx, 1]),
        'slv': float(pareto_objectives[idx, 2]),
        'method': method,
        'front_size': len(pareto_objectives)
    }


def load_race_data(project_root: Path) -> Dict:
    """Load race data."""
    race_file = project_root / "src" / "data" / "raceData.json"
    with open(race_file, 'r') as f:
        return json.load(f)


def parse_home_systems(faction_names, player_count, include_pok, include_thunders_edge, tile_db, project_root):
    """Parse home systems from faction names."""
    race_data = load_race_data(project_root)
    race_to_home_map = race_data.get("raceToHomeSystemMap", {})
    
    available_home_ids = set(race_data.get("homeSystems", []))
    if include_pok:
        available_home_ids.update(race_data.get("pokHomeSystems", []))
    if include_thunders_edge:
        available_home_ids.update(race_data.get("thundersEdgeHomeSystems", []))
    
    def normalize_home_id(home_id) -> str:
        if isinstance(home_id, str):
            try:
                home_id = int(home_id)
            except ValueError:
                return home_id
        if 1 <= home_id <= 17 or 51 <= home_id <= 58:
            return f"{home_id:02d}"
        return str(home_id)
    
    normalized_available_ids = {normalize_home_id(hid) for hid in available_home_ids}
    
    selected_home_ids = []
    for faction_name in faction_names:
        home_id = race_to_home_map.get(faction_name)
        if home_id is None:
            raise ValueError(f"Unknown faction: {faction_name}")
        if home_id not in available_home_ids:
            raise ValueError(f"Faction {faction_name} not available with current expansions")
        selected_home_ids.append(normalize_home_id(home_id))
    
    if len(selected_home_ids) < player_count:
        available_homes = [
            tile_db.tiles[tid] for tid in tile_db.home_tiles
            if tid in normalized_available_ids and tid not in selected_home_ids
        ]
        needed = player_count - len(selected_home_ids)
        additional_homes = random.sample(available_homes, needed)
        selected_home_ids.extend([str(home.id) for home in additional_homes])
    
    home_systems = []
    for home_id in selected_home_ids:
        system = tile_db.tiles.get(home_id)
        if system is None:
            raise ValueError(f"Home system {home_id} not found")
        home_systems.append(system)
    
    return home_systems


def calculate_balance_metrics(map_obj):
    evaluator = create_joebrew_evaluator()
    home_values = get_home_values(map_obj, evaluator)
    values_array = np.array([hv.value for hv in home_values])
    jaines = jains_fairness_index(values_array)
    slv_value = compute_slice_level_variance(map_obj, evaluator)
    return jaines, slv_value


def export_human_readable(map_obj, output_file, jaines, slv):
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TI4 MAP SETUP\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Jaine's Index: {jaines:.3f}\n")
        f.write(f"SLV: {slv:.2f}\n\n")
        
        tiles_by_ring = {}
        for space in map_obj.spaces:
            q, r = hexcoord_to_axial(space.coord)
            ring = max(abs(q), abs(r), abs(-q - r))
            
            if ring not in tiles_by_ring:
                tiles_by_ring[ring] = []
            
            if space.space_type == MapSpaceType.WARP:
                tile_name = f"Hyperlane {space.hyperlane_tile_id}" if space.hyperlane_tile_id else "EMPTY"
            elif space.system is None:
                tile_name = "EMPTY"
            else:
                tile_name = f"Tile #{space.system.id}"
                if space.system.planets:
                    planets = ", ".join(p.name for p in space.system.planets)
                    tile_name += f" ({planets})"

            tiles_by_ring[ring].append({
                'q': q, 'r': r,
                'tile_name': tile_name,
                'is_home': space.space_type.value == 'home'
            })

        for ring in sorted(tiles_by_ring.keys()):
            f.write(f"RING {ring}\n")
            f.write("-" * 70 + "\n")
            for tile in sorted(tiles_by_ring[ring], key=lambda t: (t['q'], t['r'])):
                home = " [HOME]" if tile['is_home'] else ""
                f.write(f"  ({tile['q']:2d}, {tile['r']:2d}): {tile['tile_name']}{home}\n")
            f.write("\n")


def export_ttpg_string(map_obj, output_file):
    tiles = [str(space.system.id if space.system else -1) for space in map_obj.spaces]
    with open(output_file, 'w') as f:
        f.write("TTPG String:\n")
        f.write("=" * 70 + "\n\n")
        f.write(" ".join(tiles) + "\n")


def main():
    parser = argparse.ArgumentParser(description="TI4 Map Generator (FIXED)")
    parser.add_argument("--players", "-p", type=int, default=6, choices=[2,3,4,5,6,7,8])
    parser.add_argument("--template", "-t", type=str, default=None)
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--no-pok", action="store_true")
    parser.add_argument("--no-thunders-edge", action="store_true")
    parser.add_argument("--uncharted", action="store_true")
    parser.add_argument("--factions", nargs="+", type=str, default=[])
    parser.add_argument(
        "--selection",
        choices=["min_slv", "min_gap", "knee", "balanced"],
        default="min_slv",
        help="Pareto selection: min_slv (local fairness), min_gap (global), knee (compromise), balanced (Tchebyshev)"
    )
    parser.add_argument(
        "--optimizer",
        choices=["g3-d", "g1"],
        default="g3-d"
    )
    
    args = parser.parse_args()
    
    # Setup
    script_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / "output" / f"game_night_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    include_pok = not args.no_pok
    include_thunders_edge = not args.no_thunders_edge
    template_name = args.template or ("warp" if args.players in [5, 7] else "normal")
    seed = args.seed or np.random.randint(1, 1000000)
    
    print("=" * 70)
    print("TI4 MAP GENERATOR (FIXED)")
    print("=" * 70)
    print(f"\nPlayers: {args.players}, Template: {template_name}")
    print(f"Optimizer: {args.optimizer}, Selection: {args.selection}")
    print(f"Output: {output_dir}\n")
    
    # Step 1: Generate
    print("[1/5] Generating map...")
    project_root = Path(__file__).parent.parent
    from ti4_analysis.data.tile_loader import load_tile_database
    tile_db = load_tile_database(project_root=project_root)
    
    home_systems = None
    if args.factions:
        home_systems = parse_home_systems(
            args.factions, args.players, include_pok,
            include_thunders_edge, tile_db, project_root
        )
    
    map_obj = generate_random_map(
        player_count=args.players,
        template_name=template_name,
        include_pok=include_pok,
        include_thunders_edge=include_thunders_edge,
        include_uncharted=args.uncharted,
        home_systems=home_systems,
        random_seed=seed,
        tile_db=tile_db,
        project_root=project_root
    )
    print(f"✓ {len(map_obj.spaces)} spaces\n")
    
    # Step 2: Initial metrics
    print("[2/5] Initial metrics...")
    evaluator = create_joebrew_evaluator()
    jaines_before, slv_before = calculate_balance_metrics(map_obj)
    gap_before = get_balance_gap(get_home_values(map_obj, evaluator))
    print(f"Before: Gap={gap_before:.1f}, Jain's={jaines_before:.3f}, SLV={slv_before:.1f}\n")
    
    # Step 3: Optimize
    if args.optimizer == "g3-d":
        print("[3/5] Running NSGA-II (~11 min)...")
        result = optimize_map_nsga2(
            ti4_map=map_obj,
            evaluator=evaluator,
            population_size=100,
            n_generations=80,
            swap_anomalies=True,
            objective_type="dispersed",
            random_seed=seed,
            verbose=True
        )
        
        print(f"\n[4/5] Selecting from Pareto front using '{args.selection}'...")
        map_obj, idx, metrics = select_from_pareto_front(
            result['pareto_front_objectives'],
            result['pareto_front_maps'],
            method=args.selection
        )
        
        print(f"✓ Selected {idx+1}/{metrics['front_size']}")
        print(f"  Gap: {metrics['gap']:.1f}, SLV: {metrics['slv']:.1f}, Moran's I: {metrics['moran']:.3f}\n")
        
    else:  # G1
        print("[3/5] Running hill-climbing (~1 min)...")
        from ti4_analysis.algorithms.balance_engine import improve_balance
        gap, _ = improve_balance(
            ti4_map=map_obj,
            evaluator=evaluator,
            iterations=1000,
            swap_anomalies=False,
            random_seed=seed,
            show_progress=True
        )
        print(f"\n✓ Final Gap: {gap:.1f}\n")
        print("[4/5] Skipping Pareto selection (G1 is single-objective)\n")
    
    # Step 5: Final metrics
    print("[5/5] Final metrics...")
    jaines_final, slv_final = calculate_balance_metrics(map_obj)
    gap_final = get_balance_gap(get_home_values(map_obj, evaluator))
    moran_final = resource_clustering_coefficient(map_obj, evaluator, include_wormholes=True)
    
    print(f"Final: Gap={gap_final:.1f}, Jain's={jaines_final:.3f}, SLV={slv_final:.1f}, Moran's I={moran_final:+.3f}\n")
    
    # Export
    print("Exporting...")
    ti4proj_file = output_dir / "game_map.ti4proj"
    export_to_ti4proj(
        map_data=map_obj,
        filename=ti4proj_file,
        map_name=f"Game Night {datetime.now().strftime('%Y-%m-%d')}",
        description=f"{args.players}p, Jain's={jaines_final:.3f}, SLV={slv_final:.1f}",
        jaines_index=jaines_final,
        metadata_extras={"slv": slv_final, "seed": seed}
    )
    
    readable_file = output_dir / "setup_instructions.txt"
    export_human_readable(map_obj, readable_file, jaines_final, slv_final)
    
    ttpg_file = output_dir / "ttpg_string.txt"
    export_ttpg_string(map_obj, ttpg_file)
    
    print("\n" + "=" * 70)
    print("✓ COMPLETE")
    print("=" * 70)
    print(f"\n1. UI:    {ti4proj_file}")
    print(f"2. Setup: {readable_file}")
    print(f"3. TTS:   {ttpg_file}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
