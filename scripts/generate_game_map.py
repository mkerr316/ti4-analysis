#!/usr/bin/env python3
"""
TI4 MAP GENERATOR - GAME NIGHT EDITION
=======================================

Generates a balanced TI4 map using the best optimizer from comprehensive analysis (G3-D/NSGA-II).

Outputs in THREE formats:
1. .ti4proj file (for UI if it works)
2. Human-readable text file (for manual board setup)
3. TTPG string (for Tabletop Simulator)

Usage:
    python generate_game_map.py [--players N] [--template NAME] [--seed N] [--optimizer g3-d|g1] [--no-pok] [--no-thunders-edge] [--uncharted]

Optimizers:
    g3-d (default): NSGA-II multi-objective optimizer - best quality (~10-15 min)
    g1: Hill-climbing single-objective - faster but lower quality (~1 min)

Outputs to: output/game_night_YYYYMMDD_HHMMSS/
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
from ti4_analysis.algorithms.nsga2_optimizer import (
    optimize_map_nsga2,
    get_knee_point
)
from ti4_analysis.spatial_stats.spatial_metrics import (
    jains_fairness_index,
    morans_i,
    create_adjacency_weights,
    SpatialWeightMatrix,
    resource_clustering_coefficient,
    compute_slice_level_variance
)
from ti4_analysis.data.map_structures import MapSpaceType, System
from typing import Dict, List, Tuple
import numpy as np
import random


def select_pareto_solution(
    pareto_objectives: np.ndarray,
    pareto_maps: List[TI4Map],
    weights: np.ndarray = np.array([0.2, 0.2, 0.6]),
    rho: float = 0.05
) -> Tuple[TI4Map, int, Dict]:
    """
    Select solution from Pareto front using Tchebyshev achievement scalarizing function.
    
    Args:
        pareto_objectives: Nx3 array of objectives [Gap, Moran's I, SLV]
        pareto_maps: List of TI4Map objects corresponding to objectives
        weights: Tchebyshev weights [Gap, Moran's I, SLV] (default: [0.2, 0.2, 0.6])
        rho: Augmentation parameter (default: 0.05)
    
    Returns:
        (selected_map, selected_index, metrics_dict)
    """
    # Normalize objectives to [0, 1]
    normalized = (pareto_objectives - pareto_objectives.min(axis=0)) / (
        pareto_objectives.max(axis=0) - pareto_objectives.min(axis=0) + 1e-8
    )
    
    # Tchebyshev achievement scalarizing function
    # max(w_i * f_i) + rho * sum(f_i)
    scores = np.max(weights * normalized, axis=1) + rho * np.sum(normalized, axis=1)
    idx = np.argmin(scores)
    
    return pareto_maps[idx], idx, {
        'gap': float(pareto_objectives[idx, 0]),
        'moran': float(pareto_objectives[idx, 1]),
        'slv': float(pareto_objectives[idx, 2]),
        'method': 'tchebyshev',
        'weights': weights.tolist(),
        'front_size': len(pareto_objectives)
    }


def meets_quality_criteria(balance_gap: float, jaines_index: float, slv: float, target: str = "good") -> bool:
    """
    Check if metrics meet quality criteria.
    
    Priority: Jaine's Index is the primary metric (most important for fairness).
    SLV (Slice-Level Variance) measures local fairness within each player's slice.
    
    Target levels (from comprehensive analysis):
    - "comprehensive": Matches comprehensive analysis G3-D results
      - Jaine's Index >= 0.999 (mean: ~0.9992)
      - SLV <= 650 (mean: 618.4, range: 598.6-632.9)
    - "excellent": Jaine's > 0.99, SLV <= 800
    - "good": Jaine's > 0.95, SLV <= 1000
    """
    if target == "comprehensive":
        # Comprehensive: Match G3-D from comprehensive analysis
        # Mean Jaine's Index: ~0.9992 (all trials >= 0.999)
        # Mean SLV: 618.4 (range: 598.6-632.9)
        return jaines_index >= 0.999 and slv <= 650
    elif target == "excellent":
        # Excellent: Jaine's Index must be excellent, SLV should be reasonable
        return jaines_index > 0.99 and slv <= 800
    else:  # "good"
        # Good: Jaine's Index must be good, SLV should be acceptable
        return jaines_index > 0.95 and slv <= 1000


def load_race_data(project_root: Path) -> Dict:
    """Load race data to map faction names to home system IDs."""
    race_file = project_root / "src" / "data" / "raceData.json"
    if not race_file.exists():
        raise FileNotFoundError(f"Race data file not found: {race_file}")
    
    with open(race_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_home_systems(
    faction_names: List[str],
    player_count: int,
    include_pok: bool,
    include_thunders_edge: bool,
    tile_db,
    project_root: Path
) -> List[System]:
    """
    Parse faction names to home system objects.
    
    If fewer factions are specified than players, randomly selects the rest.
    
    Args:
        faction_names: List of faction names (e.g., ["The Federation of Sol", "The Mentak Coalition"])
        player_count: Total number of players
        include_pok: Whether PoK factions are available
        include_thunders_edge: Whether Thunder's Edge factions are available
        tile_db: Tile database
        project_root: Project root directory
    
    Returns:
        List of System objects for home systems
    """
    race_data = load_race_data(project_root)
    race_to_home_map = race_data.get("raceToHomeSystemMap", {})
    
    # Get available home systems based on expansions
    available_home_ids = set(race_data.get("homeSystems", []))
    if include_pok:
        available_home_ids.update(race_data.get("pokHomeSystems", []))
    if include_thunders_edge:
        available_home_ids.update(race_data.get("thundersEdgeHomeSystems", []))
    
    # Convert faction names to home system IDs
    # Note: Tile database uses zero-padded strings for home systems (01-17, 51-58, etc.)
    def normalize_home_id(home_id) -> str:
        """Normalize home system ID to match tile database format (zero-padded for 1-17, 51-58)."""
        if isinstance(home_id, str):
            # Already a string, try to parse it
            try:
                home_id = int(home_id)
            except ValueError:
                return home_id  # Return as-is if not numeric
        if 1 <= home_id <= 17 or 51 <= home_id <= 58:
            return f"{home_id:02d}"  # Zero-pad to 2 digits
        return str(home_id)
    
    # Normalize available_home_ids to match tile database format
    normalized_available_ids = {normalize_home_id(hid) for hid in available_home_ids}
    
    selected_home_ids = []
    for faction_name in faction_names:
        home_id = race_to_home_map.get(faction_name)
        if home_id is None:
            raise ValueError(f"Unknown faction: {faction_name}")
        if home_id not in available_home_ids:
            raise ValueError(
                f"Faction {faction_name} (home system {home_id}) not available "
                f"with current expansion settings"
            )
        selected_home_ids.append(normalize_home_id(home_id))
    
    # If fewer factions specified than players, randomly select the rest
    if len(selected_home_ids) < player_count:
        # Get all available home systems
        available_homes = [
            tile_db.tiles[tid] for tid in tile_db.home_tiles
            if tid in normalized_available_ids and tid not in selected_home_ids
        ]
        
        needed = player_count - len(selected_home_ids)
        if len(available_homes) < needed:
            raise ValueError(
                f"Not enough available home systems. "
                f"Selected {len(selected_home_ids)}, need {needed} more, "
                f"but only {len(available_homes)} available."
            )
        
        # Randomly select the remaining home systems
        additional_homes = random.sample(available_homes, needed)
        selected_home_ids.extend([str(home.id) for home in additional_homes])
    
    # Convert IDs to System objects
    home_systems = []
    for home_id in selected_home_ids:
        system = tile_db.tiles.get(home_id)
        if system is None:
            raise ValueError(f"Home system {home_id} not found in tile database")
        home_systems.append(system)
    
    return home_systems


def verify_tile_consistency(map_obj: TI4Map, tile_db) -> List[str]:
    """
    Verify that all systems in the map have consistent tile IDs and planet data.
    
    Returns:
        List of error messages (empty if all tiles are consistent)
    """
    errors = []
    
    for space in map_obj.spaces:
        if space.space_type == MapSpaceType.SYSTEM and space.system:
            tile_id = space.system.id
            expected_system = tile_db.tiles.get(str(tile_id))
            
            if not expected_system:
                errors.append(f"Position {space.coord}: Tile #{tile_id} not found in database")
                continue
            
            # Verify planets match
            expected_planet_names = sorted([p.name for p in expected_system.planets])
            actual_planet_names = sorted([p.name for p in space.system.planets])
            
            if expected_planet_names != actual_planet_names:
                errors.append(
                    f"Position {space.coord}: Tile #{tile_id} planet mismatch - "
                    f"Expected: {expected_planet_names}, Got: {actual_planet_names}"
                )
    
    return errors


def calculate_balance_metrics(map_obj: TI4Map) -> tuple[float, float]:
    """
    Calculate real balance metrics using spatial analysis.

    Returns:
        - Jain's Fairness Index: Range [1/n, 1.0], where 1.0 = perfectly fair
        - SLV: Slice-Level Variance, lower = more fair local distribution
    """
    # Create evaluator for calculating home values
    evaluator = create_joebrew_evaluator()

    # Get home values for all player positions
    home_values = get_home_values(map_obj, evaluator)
    values_array = np.array([hv.value for hv in home_values])

    # Calculate Jain's Fairness Index
    # Formula: (Σx)² / (n × Σx²)
    jaines = jains_fairness_index(values_array)

    # Calculate SLV (Slice-Level Variance) for local fairness
    # SLV measures variance in slice totals for each player, indicating local fairness
    # Lower SLV = more fair distribution within each player's accessible slices
    slv_value = compute_slice_level_variance(map_obj, evaluator)

    return jaines, slv_value


def export_human_readable(map_obj: TI4Map, output_file: Path, jaines: float, slv: float):
    """
    Export a human-readable text file for manual board setup.

    This is your BACKUP if the UI doesn't work - you can set up the
    physical board directly from this file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("TI4 MAP SETUP - GAME NIGHT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Balance Metrics:\n")
        f.write(f"  - Jaine's Index: {jaines:.3f}\n")
        f.write(f"  - SLV: {slv:.2f}\n")
        f.write("\n" + "=" * 70 + "\n\n")

        # Group tiles by ring for easier setup
        tiles_by_ring = {}
        for space in map_obj.spaces:
            q, r = hexcoord_to_axial(space.coord)

            # Calculate ring (distance from center)
            ring = max(abs(q), abs(r), abs(-q - r))

            if ring not in tiles_by_ring:
                tiles_by_ring[ring] = []

            # Handle different space types
            tile_id = None
            if space.space_type == MapSpaceType.WARP:
                # Hyperlane tile
                if space.hyperlane_tile_id:
                    tile_name = f"Hyperlane {space.hyperlane_tile_id}"
                    if space.hyperlane_rotation is not None and space.hyperlane_rotation != 0:
                        tile_name += f" (rot: {space.hyperlane_rotation})"
                    tile_id = space.hyperlane_tile_id
                else:
                    tile_name = "EMPTY"
            elif space.system is None:
                tile_name = "EMPTY"
            else:
                tile_id = space.system.id
                tile_name = f"Tile #{tile_id}"
                
                # Always show planet information if available, or indicate empty/anomaly system
                if space.system.planets:
                    planet_names = ", ".join(p.name for p in space.system.planets)
                    tile_name += f" ({planet_names})"
                elif space.system.anomalies:
                    # Red tile with anomalies but no planets
                    anomaly_names = ", ".join(a.value.replace('_', ' ').title() for a in space.system.anomalies)
                    tile_name += f" ({anomaly_names})"
                elif space.system.wormhole:
                    # Empty system with wormhole
                    tile_name += f" (Empty - {space.system.wormhole.value.upper()} Wormhole)"
                else:
                    # Empty red tile
                    tile_name += " (Empty System)"

            tiles_by_ring[ring].append({
                'q': q, 'r': r,
                'tile_id': tile_id,
                'tile_name': tile_name,
                'is_home': space.space_type.value == 'home'
            })

        # Output by ring
        for ring in sorted(tiles_by_ring.keys()):
            f.write(f"RING {ring} {'(CENTER)' if ring == 0 else ''}\n")
            f.write("-" * 70 + "\n")

            for tile in sorted(tiles_by_ring[ring], key=lambda t: (t['q'], t['r'])):
                home_marker = " [HOME SYSTEM]" if tile['is_home'] else ""
                f.write(f"  Position ({tile['q']:2d}, {tile['r']:2d}): {tile['tile_name']}{home_marker}\n")

            f.write("\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("SETUP INSTRUCTIONS:\n")
        f.write("=" * 70 + "\n\n")
        # Find Mecatol Rex tile ID from the map
        mecatol_id = None
        for space in map_obj.spaces:
            if space.system and space.system.is_mecatol_rex():
                mecatol_id = space.system.id
                break
        if mecatol_id:
            f.write(f"1. Place Mecatol Rex (Tile #{mecatol_id}) in the center\n")
        else:
            f.write("1. Place Mecatol Rex in the center\n")
        f.write("2. Work outward ring by ring\n")
        f.write("3. Use the (q, r) coordinates to determine positions:\n")
        f.write("   - (0, 0) = Center\n")
        f.write("   - Positive q = East direction\n")
        f.write("   - Positive r = Southeast direction\n")
        f.write("4. Home systems are marked with [HOME SYSTEM]\n")
        f.write("\nEnjoy your game!\n")


def export_ttpg_string(map_obj: TI4Map, output_file: Path):
    """
    Export TTPG (Tabletop Playground/Simulator) string format.

    This can be pasted into TTS for online play.
    """
    # Build tile array in position index order (0-36 for base, up to 60 for PoK)
    # This is a simplified version - you may need to adjust based on your
    # exact TTPG format

    tile_string = ""

    # Sort spaces by position index (derived from coordinates)
    # For now, just output the tile IDs in coordinate order
    tiles = []
    for space in map_obj.spaces:
        tile_id = space.system.id if space.system else -1
        tiles.append(str(tile_id))

    tile_string = " ".join(tiles)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TTPG String (paste into Tabletop Simulator):\n")
        f.write("=" * 70 + "\n\n")
        f.write(tile_string + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("Note: This is a simplified format. Adjust based on your\n")
        f.write("      exact TTPG mod requirements.\n")


def get_default_template(player_count: int) -> str:
    """
    Get default template name for a given player count.
    
    Args:
        player_count: Number of players (2-8)
    
    Returns:
        Default template name
    """
    defaults = {
        2: "normal",
        3: "normal",
        4: "normal",
        5: "warp",  # 5-player uses hyperlanes by default
        6: "normal",
        7: "warp",  # 7-player uses hyperlanes by default
        8: "normal",
    }
    return defaults.get(player_count, "normal")


def main():
    """Generate map and export in multiple formats."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate a balanced TI4 map for game night",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_game_map.py --players 5
  python generate_game_map.py --players 6 --template spiral --seed 123
  python generate_game_map.py --players 5 --no-pok --no-thunders-edge
  python generate_game_map.py --players 4 --uncharted
  python generate_game_map.py --players 5 --factions "The Federation of Sol" "The Mentak Coalition"
  python generate_game_map.py --players 6 --factions "The Federation of Sol" --swap-homes
        """
    )
    parser.add_argument(
        "--players", "-p",
        type=int,
        default=6,
        choices=[2, 3, 4, 5, 6, 7, 8],
        help="Number of players (default: 6)"
    )
    parser.add_argument(
        "--template", "-t",
        type=str,
        default=None,
        help="Template name (default: 'warp' for 5/7 players, 'normal' otherwise)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)"
    )
    parser.add_argument(
        "--no-pok",
        action="store_true",
        help="Exclude Prophecy of Kings expansion (default: include PoK)"
    )
    parser.add_argument(
        "--no-thunders-edge",
        action="store_true",
        help="Exclude Thunder's Edge expansion (default: include Thunder's Edge)"
    )
    parser.add_argument(
        "--uncharted",
        action="store_true",
        help="Include Uncharted Space expansion (default: exclude)"
    )
    parser.add_argument(
        "--factions", "--homes",
        nargs="+",
        type=str,
        default=[],
        metavar="FACTION",
        help="Specify home systems by faction name. "
             "If fewer factions specified than players, randomly selects the rest. "
             "Faction names must match exactly (e.g., 'The Federation of Sol', 'The Mentak Coalition'). "
             "Use quotes for multi-word names. Example: --factions 'The Federation of Sol' 'The Mentak Coalition'"
    )
    parser.add_argument(
        "--swap-homes",
        action="store_true",
        help="Allow home systems to be swapped with other home systems during optimization "
             "(improves balance by finding better home positions)"
    )
    parser.add_argument(
        "--target",
        choices=["good", "excellent", "comprehensive"],
        default="comprehensive",
        help="Target quality level: 'comprehensive' (Jaine's>=0.999, matches G3-D analysis), "
             "'excellent' (Jaine's>0.99), or 'good' (Jaine's>0.95). "
             "Only used with --retry flag (default: comprehensive)"
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Enable retry logic - will retry until quality criteria are met (default: disabled, generates one map)"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=50,
        help="Maximum number of generation attempts before giving up (only used with --retry, default: 50)"
    )
    parser.add_argument(
        "--optimizer",
        choices=["g3-d", "g1"],
        default="g3-d",
        help="Optimization algorithm: 'g3-d' (NSGA-II multi-objective, best quality, ~10-15 min) "
             "or 'g1' (hill-climbing single-objective, faster, ~1 min). Default: g3-d"
    )
    
    args = parser.parse_args()
    
    # Determine template
    template_name = args.template
    if template_name is None:
        template_name = get_default_template(args.players)

    print("=" * 70, flush=True)
    print("TI4 MAP GENERATOR - GAME NIGHT EDITION", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    
    # Display quality target if retry is enabled
    if args.retry:
        print(f"Retry mode: ENABLED", flush=True)
        print(f"Target: {args.target.upper()} quality", flush=True)
        if args.target == "comprehensive":
            print(f"  - Jaine's Index: >= 0.999 (matches comprehensive analysis G3-D)", flush=True)
            print(f"  - SLV: <= 650 (comprehensive analysis mean: 618.4)", flush=True)
            print(f"  - Balance Gap: Not a hard requirement (comprehensive analysis mean: 41.6)", flush=True)
        elif args.target == "excellent":
            print(f"  - Jaine's Index: > 0.99 (PRIMARY - most important)", flush=True)
            print(f"  - SLV: <= 800", flush=True)
        else:  # "good"
            print(f"  - Jaine's Index: > 0.95 (PRIMARY - most important)", flush=True)
            print(f"  - SLV: <= 1000", flush=True)
        print(f"  - Balance Gap: Not a hard requirement (improves naturally with good Jaine's Index)", flush=True)
        print(f"  - Max attempts: {args.max_attempts}", flush=True)
        print(flush=True)
    else:
        print(f"Retry mode: DISABLED (generating one map)", flush=True)
        print(flush=True)
    
    # Retry loop
    best_gap = float('inf')
    best_output_dir = None
    best_attempt = 0
    
    # Determine max attempts: 1 if retry disabled, otherwise use --max-attempts
    max_attempts = 1 if not args.retry else args.max_attempts
    
    for attempt in range(1, max_attempts + 1):
        if args.retry and attempt > 1:
            print(f"\n{'=' * 70}", flush=True)
            print(f"ATTEMPT {attempt}/{max_attempts}", flush=True)
            print(f"{'=' * 70}", flush=True)
            print(flush=True)
        
        # Determine seed for this attempt (use provided seed only on first attempt)
        if args.seed is None or attempt > 1:
            import random as random_module
            seed = random_module.randint(1, 1000000)
        else:
            seed = args.seed
        
        # Create output directory with timestamp (relative to script location)
        script_dir = Path(__file__).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / "output" / f"game_night_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if attempt == 1 or not args.retry:
            print(f"Output directory: {output_dir.absolute()}", flush=True)
            print(flush=True)

        # Step 1: Generate map
        # Determine expansion settings
        include_pok = not args.no_pok
        include_thunders_edge = not args.no_thunders_edge
        include_uncharted = args.uncharted
        
        print("Step 1: Generating map...", flush=True)
        print(f"  - Players: {args.players}", flush=True)
        print(f"  - Template: {template_name}", flush=True)
        expansions_list = []
        if include_pok:
            expansions_list.append("PoK")
        if include_thunders_edge:
            expansions_list.append("Thunder's Edge")
        if include_uncharted:
            expansions_list.append("Uncharted Space")
        expansions_str = ", ".join(expansions_list) if expansions_list else "Base only"
        print(f"  - Expansions: {expansions_str}", flush=True)
        print(f"  - Random seed: {seed}", flush=True)
        if args.swap_homes:
            print(f"  - Home system swapping: ENABLED (will optimize home positions)", flush=True)
        print(flush=True)

        # Find project root (parent of ti4-analysis directory)
        project_root = Path(__file__).parent.parent
        
        # Load tile database for home system selection
        from ti4_analysis.data.tile_loader import load_tile_database
        tile_db = load_tile_database(project_root=project_root)
        
        # Parse home systems if specified
        home_systems = None
        if args.factions:
            print(f"  - Specified factions: {', '.join(args.factions)}", flush=True)
            try:
                home_systems = parse_home_systems(
                    faction_names=args.factions,
                    player_count=args.players,
                    include_pok=include_pok,
                    include_thunders_edge=include_thunders_edge,
                    tile_db=tile_db,
                    project_root=project_root
                )
                print(f"  - Selected {len(home_systems)} home systems", flush=True)
                if len(args.factions) < args.players:
                    print(f"  - Randomly selected {args.players - len(args.factions)} additional home systems", flush=True)
            except Exception as e:
                print(f"  [ERROR] Failed to parse factions: {e}", flush=True, file=sys.stderr)
                raise
            print(flush=True)

        map_obj = generate_random_map(
            player_count=args.players,
            template_name=template_name,
            include_pok=include_pok,
            include_thunders_edge=include_thunders_edge,
            include_uncharted=include_uncharted,
            home_systems=home_systems,
            random_seed=seed,
            tile_db=tile_db,
            project_root=project_root
        )

        # Count hyperlanes
        hyperlane_count = sum(1 for s in map_obj.spaces if s.space_type == MapSpaceType.WARP)
        
        # Get home system info
        home_spaces = map_obj.get_home_spaces()
        
        print(f"[OK] Generated map with {len(map_obj.spaces)} spaces", flush=True)
        if hyperlane_count > 0:
            print(f"  - Hyperlanes: {hyperlane_count}", flush=True)
        if home_systems:
            print(f"  - Home systems: {len(home_spaces)} factions", flush=True)
            # Try to get faction names for display
            try:
                race_data = load_race_data(project_root)
                home_to_race_map = race_data.get("homeSystemToRaceMap", {})
                faction_names = []
                for home_space in home_spaces:
                    if home_space.system:
                        race_name = home_to_race_map.get(str(home_space.system.id), f"Tile {home_space.system.id}")
                        faction_names.append(race_name)
                if faction_names:
                    print(f"    Factions: {', '.join(faction_names)}", flush=True)
            except:
                pass  # If we can't load race data, just skip this
        print(flush=True)

        # Step 2: Calculate initial metrics
        print("Step 2: Calculating initial balance metrics...", flush=True)
        evaluator = create_joebrew_evaluator()
        jaines_before, slv_before = calculate_balance_metrics(map_obj)
        initial_home_values = get_home_values(map_obj, evaluator)
        balance_gap_before = get_balance_gap(initial_home_values)
        morans_i_before = resource_clustering_coefficient(map_obj, evaluator, include_wormholes=True)
        print(f"  - Balance Gap (before): {balance_gap_before:.2f}", flush=True)
        print(f"  - Jaine's Index (before): {jaines_before:.3f}", flush=True)
        print(f"  - SLV (before): {slv_before:.2f}", flush=True)
        print(f"  - Moran's I (before): {morans_i_before:+.3f} (diagnostic)", flush=True)
        print(flush=True)

        # Step 3: Optimize balance
        
        if args.optimizer == "g3-d":
            # Use NSGA-II (G3-D) - best quality from comprehensive analysis
            print("Step 3: Optimizing map balance...", flush=True)
            print("  - Running NSGA-II multi-objective optimizer (G3-D)", flush=True)
            print("  - Population: 100, Generations: 80 (8000 evaluation budget)", flush=True)
            print("  - Objectives: [Balance Gap, Moran's I, Slice-Level Variance]", flush=True)
            print("  - Quality criteria: Jaine's Index >= 0.999, SLV <= 650", flush=True)
            print("  - Objective type: 'dispersed' (matches comprehensive analysis G3-D)", flush=True)
            print("  - This may take 10-15 minutes...", flush=True)
            print(flush=True)

            # Run NSGA-II optimization (G3-D: unconstrained action space)
            # Configuration matches comprehensive analysis: 8000 evaluations, dispersed objective
            optimization_result = optimize_map_nsga2(
                ti4_map=map_obj,
                evaluator=evaluator,
                population_size=100,
                n_generations=80,  # 8000 evaluations ÷ 100 population = 80 generations (matches comprehensive analysis)
                swap_anomalies=True,  # Unconstrained action space (G3-D)
                objective_type="dispersed",  # Matches G3-D from comprehensive analysis (minimizes Moran's I directly)
                random_seed=seed,
                verbose=True
            )
            
            # Extract Pareto front
            pareto_objectives = optimization_result['pareto_front_objectives']
            pareto_maps = optimization_result['pareto_front_maps']
            
            # Select best solution using Tchebyshev scalarization with weights [0.2, 0.2, 0.6]
            # This prioritizes SLV (local fairness) while still considering Gap and Moran's I
            # Matches RESEARCH_PLAN.md specification
            if len(pareto_objectives) > 0:
                weights = np.array([0.2, 0.2, 0.6])  # [Gap, Moran's I, SLV]
                optimized_map, best_idx, selection_metrics = select_pareto_solution(
                    pareto_objectives=pareto_objectives,
                    pareto_maps=pareto_maps,
                    weights=weights,
                    rho=0.05
                )
                balance_gap = selection_metrics['gap']
                print(f"[OK] Selected solution from Pareto front using Tchebyshev scalarization", flush=True)
                print(f"  - Weights: Gap={weights[0]}, Moran's I={weights[1]}, SLV={weights[2]}", flush=True)
                print(f"  - Balance gap: {balance_gap:.2f}", flush=True)
                print(f"  - SLV: {selection_metrics['slv']:.2f}", flush=True)
                print(f"  - Moran's I: {selection_metrics['moran']:.3f} (diagnostic)", flush=True)
            else:
                raise RuntimeError("NSGA-II optimization produced empty Pareto front")
            
            # Update map_obj to use optimized map
            map_obj = optimized_map
            
            print(f"[OK] Balance optimization complete", flush=True)
            print(f"  - Final balance gap: {balance_gap:.2f}", flush=True)
            print(f"  - Pareto front size: {len(pareto_objectives)}", flush=True)
            print(f"  - Total evaluations: {optimization_result['n_evaluations']}", flush=True)
            if args.swap_homes:
                print(f"  - Note: Home system swapping not supported by NSGA-II (G3-D)", flush=True)
                print(f"    Home positions remain fixed during optimization", flush=True)
            print(flush=True)
        else:
            # Use G1 (hill-climbing) - faster but lower quality
            print("Step 3: Optimizing map balance...", flush=True)
            print("  - Running hill-climbing optimizer (G1)", flush=True)
            print("  - Iterations: 1000 (this may take a minute)", flush=True)
            print(flush=True)

            from ti4_analysis.algorithms.balance_engine import improve_balance
            
            balance_gap, history = improve_balance(
                ti4_map=map_obj,
                evaluator=evaluator,
                iterations=1000,
                swap_anomalies=False,
                swap_home_systems=args.swap_homes,
                random_seed=seed,
                show_progress=True
            )

            print(f"[OK] Balance optimization complete", flush=True)
            print(f"  - Final balance gap: {balance_gap:.2f}", flush=True)
            print(f"  - Swaps accepted: {history[-1].get('swaps_accepted', 'N/A')}", flush=True)
            print(flush=True)

        # Step 4: Calculate final metrics
        print("Step 4: Calculating final balance metrics...", flush=True)
        jaines_index, slv_value = calculate_balance_metrics(map_obj)
        
        # Recalculate balance gap from final map to ensure accuracy (for both optimizers)
        final_home_values = get_home_values(map_obj, evaluator)
        balance_gap = get_balance_gap(final_home_values)
        
        # Calculate Moran's I for diagnostics (not used for quality criteria)
        morans_i_value = resource_clustering_coefficient(map_obj, evaluator, include_wormholes=True)
        
        # Verify tile consistency (check that planet names match tile IDs)
        # This ensures the optimizer didn't corrupt system data during swapping
        verification_errors = verify_tile_consistency(map_obj, tile_db)
        if verification_errors:
            print(f"  [ERROR] Tile consistency check found {len(verification_errors)} errors:", flush=True)
            for error in verification_errors[:5]:  # Show first 5 errors
                print(f"    - {error}", flush=True)
            if len(verification_errors) > 5:
                print(f"    ... and {len(verification_errors) - 5} more errors", flush=True)
            print(f"  [WARNING] This indicates a bug in the optimizer - tile data may be corrupted!", flush=True)
        else:
            print(f"  [OK] Tile consistency verified - all tile IDs match their planet data", flush=True)
        
        print(f"  - Balance Gap (after): {balance_gap:.2f}", flush=True)
        print(f"  - Jaine's Index (after): {jaines_index:.3f}", flush=True)
        print(f"  - SLV (after): {slv_value:.2f}", flush=True)
        print(f"  - Moran's I (after): {morans_i_value:+.3f} (diagnostic)", flush=True)
        print(f"  - Improvement: {(jaines_index - jaines_before):.3f}", flush=True)
        print(flush=True)
        
        # Check if meets quality criteria (only relevant if retry is enabled)
        qualified = meets_quality_criteria(balance_gap, jaines_index, slv_value, args.target) if args.retry else True
        
        # If retry is disabled or map is qualified, proceed to export
        if not args.retry or qualified:
            break  # Exit retry loop to export
        else:
            # Show which criteria failed
            failed_criteria = []
            if args.target == "comprehensive":
                if jaines_index < 0.999:
                    failed_criteria.append(f"Jaine's {jaines_index:.3f} < 0.999 (target: >=0.999)")
                if slv_value > 650:
                    failed_criteria.append(f"SLV {slv_value:.2f} > 650 (target: <=650)")
            elif args.target == "excellent":
                if jaines_index <= 0.99:
                    failed_criteria.append(f"Jaine's {jaines_index:.3f} <= 0.99")
                if slv_value > 800:
                    failed_criteria.append(f"SLV {slv_value:.2f} > 800")
            else:  # "good"
                if jaines_index <= 0.95:
                    failed_criteria.append(f"Jaine's {jaines_index:.3f} <= 0.95")
                if slv_value > 1000:
                    failed_criteria.append(f"SLV {slv_value:.2f} > 1000")
            
            failed_str = ", ".join(failed_criteria) if failed_criteria else "Unknown reason"
            print(f"Attempt {attempt}: Gap={balance_gap:.1f}, Jaine's={jaines_index:.3f}, SLV={slv_value:.2f}, Moran's I={morans_i_value:+.3f} - Not qualified ({failed_str}), retrying...", flush=True)
            if balance_gap < best_gap:
                best_gap = balance_gap
                best_output_dir = output_dir
                best_attempt = attempt
            continue  # Retry
    
    # Export the map (either qualified or best found, or if retry was disabled)
    # Step 5: Export .ti4proj
    print("Step 5: Exporting .ti4proj file...", flush=True)
    ti4proj_file = output_dir / "game_map.ti4proj"
    try:
        export_to_ti4proj(
            map_data=map_obj,
            filename=ti4proj_file,
            map_name=f"Game Night {datetime.now().strftime('%Y-%m-%d')}",
            description=f"Balanced {args.players}-player map ({template_name}). Jaine's Index: {jaines_index:.3f}, SLV: {slv_value:.2f}",
            jaines_index=jaines_index,
            morans_i=None,  # No longer using Moran's I, but keeping parameter for backward compatibility
            metadata_extras={
                "generator": "Emergency CLI Generator",
                "player_count": args.players,
                "template": template_name,
                "seed": seed,
                "include_pok": include_pok,
                "include_thunders_edge": include_thunders_edge,
                "include_uncharted": include_uncharted,
                "generated_at": datetime.now().isoformat(),
                "slv": slv_value
            }
        )

        # Validate
        if validate_ti4proj(ti4proj_file):
            print(f"[OK] Valid .ti4proj file created: {ti4proj_file.absolute()}", flush=True)
        else:
            print(f"[ERROR] Warning: .ti4proj validation failed", flush=True, file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to export .ti4proj: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc()
    print(flush=True)

    # Step 6: Export human-readable
    print("Step 6: Creating human-readable setup guide...", flush=True)
    readable_file = output_dir / "setup_instructions.txt"
    try:
        export_human_readable(map_obj, readable_file, jaines_index, slv_value)
        print(f"[OK] Setup guide created: {readable_file.absolute()}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to export setup guide: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc()
    print(flush=True)

    # Step 7: Export TTPG string
    print("Step 7: Creating TTPG string...", flush=True)
    ttpg_file = output_dir / "ttpg_string.txt"
    try:
        export_ttpg_string(map_obj, ttpg_file)
        print(f"[OK] TTPG string created: {ttpg_file.absolute()}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to export TTPG string: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc()
    print(flush=True)
    
    # If we exhausted attempts without finding qualified map
    if args.retry and not qualified:
        print(flush=True)
        print("=" * 70, flush=True)
        if best_output_dir:
            print(f"Best map found (attempt {best_attempt}):", flush=True)
            print(f"  - Balance Gap: {best_gap:.2f}", flush=True)
            print(f"Output directory: {best_output_dir.absolute()}", flush=True)
            print(flush=True)
            print("No qualified map found within max attempts.", flush=True)
            print("You can:", flush=True)
            print("  1. Increase --max-attempts", flush=True)
            print("  2. Use --target good (instead of excellent)", flush=True)
            print("  3. Remove --retry flag to accept the current map", flush=True)
        else:
            print("No qualified map found. Try increasing --max-attempts.", flush=True)
        print("=" * 70, flush=True)
        print(flush=True)
    elif qualified:
        print("=" * 70, flush=True)
        print("QUALIFIED MAP FOUND!", flush=True)
        print("=" * 70, flush=True)
        print(flush=True)
        print(f"Final Metrics:", flush=True)
        print(f"  - Balance Gap: {balance_gap:.2f}", flush=True)
        print(f"  - Jaine's Index: {jaines_index:.3f}", flush=True)
        print(f"  - SLV: {slv_value:.2f}", flush=True)
        print(f"  - Moran's I: {morans_i_value:+.3f} (diagnostic)", flush=True)
        print(flush=True)

    # Summary
    print("=" * 70, flush=True)
    print("GENERATION COMPLETE!", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print("Your map has been saved in THREE formats:", flush=True)
    print(flush=True)
    print(f"1. UI Format:        {ti4proj_file.absolute()}", flush=True)
    print(f"   -> Open this in TI4 Map Analyzer (if UI works)", flush=True)
    print(flush=True)
    print(f"2. Setup Guide:      {readable_file.absolute()}", flush=True)
    print(f"   -> Read this to set up the physical board manually", flush=True)
    print(flush=True)
    print(f"3. TTS Format:       {ttpg_file.absolute()}", flush=True)
    print(f"   -> Paste this into Tabletop Simulator", flush=True)
    print(flush=True)
    print("=" * 70, flush=True)
    print("NEXT STEPS:", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print("Option A (if UI works):", flush=True)
    print(f"  1. Start the app: pnpm dev", flush=True)
    print(f"  2. File -> Open Project", flush=True)
    print(f"  3. Select: {ti4proj_file.absolute()}", flush=True)
    print(flush=True)
    print("Option B (if UI doesn't work):", flush=True)
    print(f"  1. Open: {readable_file.absolute()}", flush=True)
    print(f"  2. Follow the setup instructions", flush=True)
    print(f"  3. Place physical tiles according to the coordinates", flush=True)
    print(flush=True)
    print("Have a great game!", flush=True)
    print(flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Map generation cancelled by user.", flush=True, file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Map generation failed: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
