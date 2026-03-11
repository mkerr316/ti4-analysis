"""
Random map generator for TI4 balance experiments.

Generates randomized maps following the standard TI4 map structure
with configurable tile pools and constraints.
"""

import random
from typing import List, Optional, Dict
from pathlib import Path

from ..algorithms.hex_grid import HexCoord, get_adjacent_coordinates
from ..data.map_structures import System, MapSpace, MapSpaceType
from ..data.tile_loader import load_tile_database, load_board_template, TileDatabase
from ..algorithms.balance_engine import TI4Map


def generate_random_map(
    player_count: int = 6,
    template_name: str = "normal",
    include_pok: bool = True,
    include_uncharted: bool = False,
    home_systems: Optional[List[System]] = None,
    random_seed: Optional[int] = None,
    tile_db: Optional[TileDatabase] = None,
    project_root: Optional[Path] = None
) -> TI4Map:
    """
    Generate a random TI4 map with proper tile placement.

    This creates a fully randomized map following TI4 constraints:
    - Mecatol Rex at position 0 (center)
    - Home systems at designated positions
    - Random blue/red tiles in swappable positions
    - Proper blue-to-red ratio (60/40 for 6-player)

    Args:
        player_count: Number of players (2-8)
        template_name: Board template variant ("normal", "spiral", "large", etc.)
        include_pok: Include Prophecy of Kings expansion tiles
        include_uncharted: Include Uncharted Space expansion tiles
        home_systems: Specific home systems to use. If None, randomly selects from pool.
        random_seed: Random seed for reproducibility
        tile_db: Pre-loaded tile database (loads if None)
        project_root: Project root directory for data files

    Returns:
        TI4Map with randomized tile placement
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Load tile database
    if tile_db is None:
        tile_db = load_tile_database(project_root=project_root)

    # Load board template
    template = load_board_template(player_count, template_name, project_root)

    # Extract position lists
    home_positions = template["home_worlds"]
    swappable_positions = (
        template["primary_tiles"] +
        template["secondary_tiles"] +
        template["tertiary_tiles"]
    )

    # Get Mecatol Rex (tile 18)
    mecatol = tile_db.tiles.get("18")
    if mecatol is None:
        raise ValueError("Mecatol Rex (tile 18) not found in tile database")

    # Select home systems
    if home_systems is None:
        # Randomly select home systems from available pool
        available_homes = [
            tile_db.tiles[tid] for tid in tile_db.home_tiles
            if tid in tile_db.base_tiles or (include_pok and tid in tile_db.pok_tiles)
        ]
        if len(available_homes) < player_count:
            raise ValueError(
                f"Not enough home systems for {player_count} players. "
                f"Only {len(available_homes)} available."
            )
        home_systems = random.sample(available_homes, player_count)

    # Get swappable tile pool
    num_swappable = len(swappable_positions)
    blue_count = int(num_swappable * 0.6)  # 60% blue
    red_count = num_swappable - blue_count  # 40% red

    swappable_tiles = tile_db.get_swappable_tiles(
        include_pok=include_pok,
        include_uncharted=include_uncharted,
        blue_count=blue_count,
        red_count=red_count
    )

    # Split into red / blue based on the database definition
    red_tiles = [t for t in swappable_tiles if str(t.id) in tile_db.red_tiles]
    blue_tiles = [t for t in swappable_tiles if str(t.id) in tile_db.blue_tiles]

    if len(red_tiles) != red_count or len(blue_tiles) != blue_count:
        raise ValueError(
            f"Tile pool mismatch: expected {blue_count} blue / {red_count} red, "
            f"got {len(blue_tiles)} blue / {len(red_tiles)} red."
        )

    # Constructively assign tiles so that no red system borders another red system
    coord_to_system = _assign_swappable_tiles_without_adjacent_red(
        red_tiles=red_tiles,
        blue_tiles=blue_tiles,
        swappable_positions=swappable_positions,
    )

    # Build the map
    spaces = []

    # Position 0: Mecatol Rex
    spaces.append(MapSpace(
        coord=HexCoord(0, 0, 0),  # Center position
        space_type=MapSpaceType.SYSTEM,
        system=mecatol
    ))

    # Assign home systems to home positions
    # Note: We need to map position indices to hex coordinates
    # For now, we'll use a simplified coordinate system
    for i, (position_idx, home_system) in enumerate(zip(home_positions, home_systems)):
        coord = _position_index_to_hex_coord(position_idx)
        spaces.append(MapSpace(
            coord=coord,
            space_type=MapSpaceType.HOME,
            system=home_system
        ))

    # Assign swappable tiles to swappable positions using the
    # precomputed non-adjacent-red assignment.
    for position_idx in swappable_positions:
        coord = _position_index_to_hex_coord(position_idx)
        tile = coord_to_system.get(coord)
        if tile is None:
            raise ValueError(f"No system assigned for swappable position index {position_idx}")
        spaces.append(MapSpace(
            coord=coord,
            space_type=MapSpaceType.SYSTEM,
            system=tile
        ))

    return TI4Map(spaces)


def _position_index_to_hex_coord(index: int) -> HexCoord:
    """
    Convert a board position index (0-60) to hex coordinates.

    This mapping corresponds to the standard TI4 board layout
    with flat-top hexagonal tiles in a spiral pattern.

    Args:
        index: Position index from board template

    Returns:
        HexCoord for that position
    """
    # Standard 6-player board coordinate mapping (37 positions for base, 61 for PoK)
    # Position 0 is center (0, 0, 0)
    # Positions spiral outward in rings

    # Ring 0 (center): Position 0
    if index == 0:
        return HexCoord(0, 0, 0)

    # Ring 1: Positions 1-6
    ring1_coords = [
        HexCoord(1, -1, 0), HexCoord(1, 0, -1), HexCoord(0, 1, -1),
        HexCoord(-1, 1, 0), HexCoord(-1, 0, 1), HexCoord(0, -1, 1)
    ]
    if 1 <= index <= 6:
        return ring1_coords[index - 1]

    # Ring 2: Positions 7-18
    ring2_coords = [
        HexCoord(2, -2, 0), HexCoord(2, -1, -1), HexCoord(2, 0, -2),
        HexCoord(1, 1, -2), HexCoord(0, 2, -2), HexCoord(-1, 2, -1),
        HexCoord(-2, 2, 0), HexCoord(-2, 1, 1), HexCoord(-2, 0, 2),
        HexCoord(-1, -1, 2), HexCoord(0, -2, 2), HexCoord(1, -2, 1)
    ]
    if 7 <= index <= 18:
        return ring2_coords[index - 7]

    # Ring 3: Positions 19-36
    ring3_coords = [
        HexCoord(3, -3, 0), HexCoord(3, -2, -1), HexCoord(3, -1, -2),
        HexCoord(3, 0, -3), HexCoord(2, 1, -3), HexCoord(1, 2, -3),
        HexCoord(0, 3, -3), HexCoord(-1, 3, -2), HexCoord(-2, 3, -1),
        HexCoord(-3, 3, 0), HexCoord(-3, 2, 1), HexCoord(-3, 1, 2),
        HexCoord(-3, 0, 3), HexCoord(-2, -1, 3), HexCoord(-1, -2, 3),
        HexCoord(0, -3, 3), HexCoord(1, -3, 2), HexCoord(2, -3, 1)
    ]
    if 19 <= index <= 36:
        return ring3_coords[index - 19]

    # Ring 4 (PoK expansion): Positions 37-60
    ring4_coords = [
        HexCoord(4, -4, 0), HexCoord(4, -3, -1), HexCoord(4, -2, -2),
        HexCoord(4, -1, -3), HexCoord(4, 0, -4), HexCoord(3, 1, -4),
        HexCoord(2, 2, -4), HexCoord(1, 3, -4), HexCoord(0, 4, -4),
        HexCoord(-1, 4, -3), HexCoord(-2, 4, -2), HexCoord(-3, 4, -1),
        HexCoord(-4, 4, 0), HexCoord(-4, 3, 1), HexCoord(-4, 2, 2),
        HexCoord(-4, 1, 3), HexCoord(-4, 0, 4), HexCoord(-3, -1, 4),
        HexCoord(-2, -2, 4), HexCoord(-1, -3, 4), HexCoord(0, -4, 4),
        HexCoord(1, -4, 3), HexCoord(2, -4, 2), HexCoord(3, -4, 1)
    ]
    if 37 <= index <= 60:
        return ring4_coords[index - 37]

    raise ValueError(f"Invalid position index: {index}. Must be 0-60.")


def _assign_swappable_tiles_without_adjacent_red(
    red_tiles: List[System],
    blue_tiles: List[System],
    swappable_positions: List[int],
    max_attempts: int = 1000,
) -> Dict[HexCoord, System]:
    """
    Assign swappable tiles to board positions such that no red system
    (empty/anomaly) is adjacent to another red system.

    Uses a constructive placement with retries:
    - Place all red tiles first into positions whose neighbors are not red
    - Then fill remaining positions with blue tiles
    - Retry with different random orders if placement fails
    """
    if len(red_tiles) + len(blue_tiles) != len(swappable_positions):
        raise ValueError(
            "Number of swappable tiles does not match number of swappable positions"
        )

    coords = [_position_index_to_hex_coord(idx) for idx in swappable_positions]

    for _ in range(max_attempts):
        # Shuffle tiles and coordinates to avoid bias
        shuffled_coords = coords[:]
        random.shuffle(shuffled_coords)

        shuffled_red = red_tiles[:]
        shuffled_blue = blue_tiles[:]
        random.shuffle(shuffled_red)
        random.shuffle(shuffled_blue)

        coord_to_system: Dict[HexCoord, System] = {}
        red_coords = set()

        # Place red tiles first, enforcing no red-adjacent-to-red
        for tile in shuffled_red:
            placed = False

            # Try coordinates in random order for each tile
            random.shuffle(shuffled_coords)
            for coord in shuffled_coords:
                if coord in coord_to_system:
                    continue

                # Check if any neighbor already has a red tile
                has_red_neighbor = any(
                    neighbor in red_coords
                    for neighbor in get_adjacent_coordinates(coord)
                )
                if has_red_neighbor:
                    continue

                coord_to_system[coord] = tile
                red_coords.add(coord)
                placed = True
                break

            if not placed:
                # Failed to place this red tile without violating the constraint;
                # restart with a new random ordering.
                break

        # If we did not place all red tiles, retry
        if len(red_coords) != len(shuffled_red):
            continue

        # Fill remaining coordinates with blue tiles
        remaining_coords = [c for c in shuffled_coords if c not in coord_to_system]
        if len(remaining_coords) != len(shuffled_blue):
            # Mismatch; retry with a fresh attempt
            continue

        for coord, tile in zip(remaining_coords, shuffled_blue):
            coord_to_system[coord] = tile

        # Success: all positions assigned with no red-bordering-red
        if len(coord_to_system) == len(coords):
            return coord_to_system

    raise RuntimeError(
        "Failed to assign swappable tiles without adjacent red systems "
        f"after {max_attempts} attempts. Consider adjusting tile counts or template."
    )


def generate_multiple_maps(
    count: int,
    player_count: int = 6,
    template_name: str = "normal",
    include_pok: bool = True,
    base_seed: Optional[int] = None,
    **kwargs
) -> List[TI4Map]:
    """
    Generate multiple random maps for batch experiments.

    Each map uses a different random seed derived from base_seed.

    Args:
        count: Number of maps to generate
        player_count: Number of players per map
        template_name: Board template to use
        include_pok: Include PoK expansion
        base_seed: Base random seed (each map gets base_seed + i)
        **kwargs: Additional arguments for generate_random_map

    Returns:
        List of TI4Map objects
    """
    # Load tile database once for efficiency
    tile_db = load_tile_database()

    maps = []
    for i in range(count):
        seed = (base_seed + i) if base_seed is not None else None
        map_obj = generate_random_map(
            player_count=player_count,
            template_name=template_name,
            include_pok=include_pok,
            random_seed=seed,
            tile_db=tile_db,
            **kwargs
        )
        maps.append(map_obj)

    return maps


def get_map_statistics(ti4_map: TI4Map) -> Dict:
    """
    Get basic statistics about a map's composition.

    Args:
        ti4_map: The map to analyze

    Returns:
        Dictionary with tile counts and composition stats
    """
    stats = {
        "total_spaces": len(ti4_map.spaces),
        "home_spaces": len(ti4_map.get_home_spaces()),
        "system_spaces": len(ti4_map.get_system_spaces()),
        "blue_tiles": 0,
        "red_tiles": 0,
        "tiles_with_planets": 0,
        "tiles_with_anomalies": 0,
        "tiles_with_wormholes": 0,
        "total_planets": 0,
    }

    for space in ti4_map.get_system_spaces():
        if space.system:
            if space.system.planets:
                stats["tiles_with_planets"] += 1
                stats["total_planets"] += len(space.system.planets)
                stats["blue_tiles"] += 1
            else:
                stats["red_tiles"] += 1

            if space.system.anomalies:
                stats["tiles_with_anomalies"] += 1

            if space.system.wormhole:
                stats["tiles_with_wormholes"] += 1

    return stats
