"""
Tile database loader for TI4 map generation.

Parses the JavaScript tile database and converts it to Python data structures.
Creates a cached JSON version for faster subsequent loads.
"""

import json
import re
import chompjs
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from .map_structures import (
    System, Planet, Anomaly, Wormhole,
    PlanetTrait, TechSpecialty
)


# Anomaly and wormhole type mappings
ANOMALY_MAP = {
    "nebula": Anomaly.NEBULA,
    "gravity-rift": Anomaly.GRAVITY_RIFT,
    "asteroid-field": Anomaly.ASTEROID_FIELD,
    "supernova": Anomaly.SUPERNOVA,
}

WORMHOLE_MAP = {
    "alpha": Wormhole.ALPHA,
    "beta": Wormhole.BETA,
    "gamma": Wormhole.GAMMA,
    "delta": Wormhole.DELTA,
    "epsilon": Wormhole.EPSILON,
    "zeta": Wormhole.ZETA,
    "eta": Wormhole.ETA,
    "theta": Wormhole.THETA,
    "iota": Wormhole.IOTA,
    "kappa": Wormhole.KAPPA,
}

TRAIT_MAP = {
    "hazardous": PlanetTrait.HAZARDOUS,
    "industrial": PlanetTrait.INDUSTRIAL,
    "cultural": PlanetTrait.CULTURAL,
    None: None,
    "undefined": None,  # bare JS undefined → chompjs produces the string "undefined"
}

SPECIALTY_MAP = {
    "biotic": TechSpecialty.BIOTIC,
    "warfare": TechSpecialty.WARFARE,
    "propulsion": TechSpecialty.PROPULSION,
    "cybernetic": TechSpecialty.CYBERNETIC,
    None: None,
    "undefined": None,  # bare JS undefined → chompjs produces the string "undefined"
}


@dataclass
class TileDatabase:
    """Container for all tile data."""
    tiles: Dict[str, System]
    base_tiles: List[str]
    pok_tiles: List[str]
    uncharted_tiles: List[str]
    blue_tiles: List[str]
    red_tiles: List[str]
    home_tiles: List[str]
    hyperlane_tiles: List[str]

    def get_swappable_tiles(
        self,
        include_pok: bool = True,
        include_uncharted: bool = False,
        blue_count: int = 18,
        red_count: int = 12
    ) -> List[System]:
        """
        Get a pool of tiles suitable for random map generation.

        Args:
            include_pok: Include Prophecy of Kings expansion
            include_uncharted: Include Uncharted Space expansion
            blue_count: Number of blue (planet) tiles to include
            red_count: Number of red (empty/anomaly) tiles to include

        Returns:
            List of System objects
        """
        # Build allowed tile pool
        allowed_ids: Set[str] = set(self.base_tiles)
        if include_pok:
            allowed_ids.update(self.pok_tiles)
        if include_uncharted:
            allowed_ids.update(self.uncharted_tiles)

        # Filter to swappable tiles only (no homes, no Mecatol)
        blue_pool = [
            self.tiles[tid] for tid in self.blue_tiles
            if tid in allowed_ids
        ]
        red_pool = [
            self.tiles[tid] for tid in self.red_tiles
            if tid in allowed_ids
        ]

        # Randomly sample (caller should shuffle/seed for reproducibility)
        import random
        selected_blue = random.sample(blue_pool, min(blue_count, len(blue_pool)))
        selected_red = random.sample(red_pool, min(red_count, len(red_pool)))

        return selected_blue + selected_red


def parse_javascript_tile_data(js_file_path: Path) -> Dict:
    """
    Parse the JavaScript tileData.js file into a Python dictionary.

    Uses chompjs (AST-level JS parser) to handle mixed single/double quotes,
    trailing commas, and other JS-specific syntax that breaks JSON parsers.
    JS constant references (ANOMALIES.NEBULA, WORMHOLES.ALPHA, etc.) are
    substituted with their literal values before parsing.

    Args:
        js_file_path: Path to tileData.js

    Returns:
        Dictionary with tile data
    """
    with open(js_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Substitute JS constant references with their literal values.
    # tileData.js uses e.g. ANOMALIES.NEBULA inside the object literal;
    # chompjs cannot resolve runtime references without executing the JS.
    JS_CONSTANT_MAP = {
        "ANOMALIES.NEBULA": '"nebula"',
        "ANOMALIES.GRAVITY_RIFT": '"gravity-rift"',
        "ANOMALIES.ASTEROID_FIELD": '"asteroid-field"',
        "ANOMALIES.SUPERNOVA": '"supernova"',
        "WORMHOLES.ALPHA": '"alpha"',
        "WORMHOLES.BETA": '"beta"',
        "WORMHOLES.GAMMA": '"gamma"',
        "WORMHOLES.DELTA": '"delta"',
        "WORMHOLES.EPSILON": '"epsilon"',
        "WORMHOLES.ZETA": '"zeta"',
        "WORMHOLES.ETA": '"eta"',
        "WORMHOLES.THETA": '"theta"',
        "WORMHOLES.IOTA": '"iota"',
        "WORMHOLES.KAPPA": '"kappa"',
        "PLANET_TRAITS.HAZARDOUS": '"hazardous"',
        "PLANET_TRAITS.INDUSTRIAL": '"industrial"',
        "PLANET_TRAITS.CULTURAL": '"cultural"',
        "PLANET_TRAITS.NONE": "null",
        "TECH_SPECIALTIES.BIOTIC": '"biotic"',
        "TECH_SPECIALTIES.WARFARE": '"warfare"',
        "TECH_SPECIALTIES.PROPULSION": '"propulsion"',
        "TECH_SPECIALTIES.CYBERNETIC": '"cybernetic"',
        "TECH_SPECIALTIES.NONE": "null",
    }
    for js_const, literal in JS_CONSTANT_MAP.items():
        content = content.replace(js_const, literal)

    # Extract the tileData object literal (before the executable tileData.green = ... lines)
    match = re.search(r'const tileData = ({.*?});?\s*tileData\.green', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find tileData object in JavaScript file")

    return chompjs.parse_js_object(match.group(1))


def convert_tile_to_system(tile_id: str, tile_data: Dict) -> System:
    """
    Convert a tile data dictionary to a System object.

    Args:
        tile_id: The tile ID (string)
        tile_data: Dictionary with tile properties

    Returns:
        System object
    """
    # Convert planets
    planets = []
    for p_data in tile_data.get("planets", []):
        # Normalize JS undefined/null artifacts to None before map lookup
        trait_val = p_data.get("trait")
        if trait_val in ("undefined", "null", ""):
            trait_val = None
        spec_val = p_data.get("specialty")
        if spec_val in ("undefined", "null", ""):
            spec_val = None
        traits = [TRAIT_MAP[trait_val]] if trait_val else None
        specialty = [SPECIALTY_MAP[spec_val]] if spec_val else None

        planet = Planet(
            name=p_data["name"],
            resources=p_data["resources"],
            influence=p_data["influence"],
            traits=traits,
            tech_specialties=specialty
        )
        planets.append(planet)

    # Convert anomalies (field is always an array in tileData.js)
    anomaly_raw = tile_data.get("anomaly", [])
    if isinstance(anomaly_raw, str):
        anomaly_raw = [anomaly_raw] if anomaly_raw else []
    anomaly_list = [ANOMALY_MAP[a] for a in anomaly_raw if a in ANOMALY_MAP]
    anomalies = anomaly_list if anomaly_list else None

    # Convert wormholes (field may be a bare string or an array in tileData.js)
    wormhole_raw = tile_data.get("wormhole", [])
    if isinstance(wormhole_raw, str):
        wormhole_list = [wormhole_raw] if wormhole_raw else []
    else:
        wormhole_list = wormhole_raw
    wormhole = WORMHOLE_MAP.get(wormhole_list[0]) if wormhole_list else None

    # Create system
    return System(
        id=int(tile_id) if tile_id.isdigit() else hash(tile_id) % (10**9),  # Hash for non-numeric IDs
        planets=planets,
        anomalies=anomalies,
        wormhole=wormhole
    )


def _load_from_canonical_json(canonical: Path, cache_file: Path, use_cache: bool) -> TileDatabase:
    """Load TileDatabase from tiles_canonical.json (separate-repo layout)."""
    with open(canonical, "r") as f:
        data = json.load(f)

    tiles: Dict[str, System] = {}
    base_tiles: List[str] = []
    pok_tiles: List[str] = []
    uncharted_tiles: List[str] = []
    blue_tiles: List[str] = []
    red_tiles: List[str] = []
    home_tiles: List[str] = []
    hyperlane_tiles: List[str] = []

    for system in data["systems"]:
        tile_id = str(system["id"])
        planets = []
        for p in system.get("planets", []):
            trait_val = p.get("trait")
            spec_val = p.get("tech_specialty")
            planet = Planet(
                name=p["name"],
                resources=p["resources"],
                influence=p["influence"],
                traits=[TRAIT_MAP[trait_val]] if trait_val and trait_val in TRAIT_MAP else None,
                tech_specialties=[SPECIALTY_MAP[spec_val]] if spec_val and spec_val in SPECIALTY_MAP else None,
            )
            planets.append(planet)

        anomaly_list = [ANOMALY_MAP[a] for a in system.get("anomalies", []) if a in ANOMALY_MAP]
        wh_list = system.get("wormholes", [])
        wh = WORMHOLE_MAP[wh_list[0]] if wh_list and wh_list[0] in WORMHOLE_MAP else None

        tiles[tile_id] = System(
            id=int(tile_id) if tile_id.isdigit() else hash(tile_id) % (10**9),
            planets=planets,
            anomalies=anomaly_list if anomaly_list else None,
            wormhole=wh,
        )

        tile_type = system.get("type")
        expansion = system.get("expansion", "base")
        if tile_type == "blue":
            blue_tiles.append(tile_id)
        elif tile_type == "red":
            red_tiles.append(tile_id)

        if expansion == "base":
            base_tiles.append(tile_id)
        elif expansion == "pok":
            pok_tiles.append(tile_id)
        else:
            uncharted_tiles.append(tile_id)

    db = TileDatabase(
        tiles=tiles,
        base_tiles=base_tiles,
        pok_tiles=pok_tiles,
        uncharted_tiles=uncharted_tiles,
        blue_tiles=blue_tiles,
        red_tiles=red_tiles,
        home_tiles=home_tiles,
        hyperlane_tiles=hyperlane_tiles,
    )

    if use_cache:
        cache_data = {
            "tiles": {
                tid: {
                    "type": "blue" if tid in blue_tiles else ("red" if tid in red_tiles else "green"),
                    "planets": [
                        {
                            "name": p.name,
                            "resources": p.resources,
                            "influence": p.influence,
                            "trait": p.traits[0].value if p.traits else None,
                            "specialty": p.tech_specialties[0].value if p.tech_specialties else None,
                        }
                        for p in s.planets
                    ],
                    "anomaly": [a.value for a in s.anomalies] if s.anomalies else [],
                    "wormhole": [s.wormhole.value] if s.wormhole else [],
                }
                for tid, s in tiles.items()
            },
            "base_tiles": base_tiles,
            "pok_tiles": pok_tiles,
            "uncharted_tiles": uncharted_tiles,
            "blue_tiles": blue_tiles,
            "red_tiles": red_tiles,
            "home_tiles": home_tiles,
            "hyperlane_tiles": hyperlane_tiles,
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

    return db


def load_tile_database(
    project_root: Optional[Path] = None,
    use_cache: bool = True,
    force_reload: bool = False
) -> TileDatabase:
    """
    Load the complete tile database from JavaScript source or cache.

    Args:
        project_root: Root directory of the ti4_map_generator project.
                     If None, attempts to find it automatically.
        use_cache: If True, load from cached JSON if available
        force_reload: If True, ignore cache and reload from JavaScript

    Returns:
        TileDatabase object with all tiles
    """
    cache_file = Path(__file__).parent / "tiles_cache.json"

    if project_root is None:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "data" / "raw" / "tileData.js").exists():
                project_root = parent
                break
        if project_root is None:
            raise FileNotFoundError(
                "Could not find data/raw/tileData.js. "
                "Run scripts/update-data.sh to populate data/raw/."
            )

    js_file = project_root / "data" / "raw" / "tileData.js"

    # Check cache
    if use_cache and not force_reload and cache_file.exists():
        print(f"Loading tiles from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            cached = json.load(f)

        # Reconstruct System objects
        tiles = {
            tid: convert_tile_to_system(tid, tdata)
            for tid, tdata in cached["tiles"].items()
        }

        return TileDatabase(
            tiles=tiles,
            base_tiles=cached["base_tiles"],
            pok_tiles=cached["pok_tiles"],
            uncharted_tiles=cached["uncharted_tiles"],
            blue_tiles=cached["blue_tiles"],
            red_tiles=cached["red_tiles"],
            home_tiles=cached["home_tiles"],
            hyperlane_tiles=cached["hyperlane_tiles"]
        )

    # Parse from JavaScript
    print(f"Parsing JavaScript tile database: {js_file}")
    raw_data = parse_javascript_tile_data(js_file)

    # Load expansion lists
    with open(js_file, 'r') as f:
        content = f.read()

    # Extract tile lists using regex
    def extract_list(list_name: str) -> List[str]:
        pattern = rf'"{list_name}":\s*\[(.*?)\]'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            items_str = match.group(1)
            # Extract quoted strings
            items = re.findall(r'"([^"]+)"', items_str)
            return items
        return []

    base_tiles = extract_list("base")
    pok_tiles = extract_list("pok")
    uncharted_tiles = extract_list("uncharted")
    hyperlane_tiles = raw_data.get("hyperlanes", [])

    # Convert all tiles to System objects
    all_tiles = raw_data["all"]
    tiles = {}
    blue_tiles = []
    red_tiles = []
    home_tiles = []

    for tile_id, tile_data in all_tiles.items():
        system = convert_tile_to_system(tile_id, tile_data)
        tiles[tile_id] = system

        # Categorize
        tile_type = tile_data.get("type")
        is_special = tile_data.get("special", False)

        if tile_type == "green" and not is_special:
            home_tiles.append(tile_id)
        elif tile_type == "blue" and not is_special:
            blue_tiles.append(tile_id)
        elif tile_type == "red" and not is_special:
            red_tiles.append(tile_id)

    db = TileDatabase(
        tiles=tiles,
        base_tiles=base_tiles,
        pok_tiles=pok_tiles,
        uncharted_tiles=uncharted_tiles,
        blue_tiles=blue_tiles,
        red_tiles=red_tiles,
        home_tiles=home_tiles,
        hyperlane_tiles=hyperlane_tiles
    )

    # Save cache
    if use_cache:
        print(f"Saving tile cache: {cache_file}")
        cache_data = {
            "tiles": {
                tid: {
                    "type": "blue" if tid in blue_tiles else ("red" if tid in red_tiles else "green"),
                    "planets": [
                        {
                            "name": p.name,
                            "resources": p.resources,
                            "influence": p.influence,
                            "trait": p.traits[0].value if p.traits else None,
                            "specialty": p.tech_specialties[0].value if p.tech_specialties else None,
                            "legendary": False  # Simplified for now
                        }
                        for p in system.planets
                    ],
                    "anomaly": [a.value for a in system.anomalies] if system.anomalies else [],
                    "wormhole": [system.wormhole.value] if system.wormhole else []
                }
                for tid, system in tiles.items()
            },
            "base_tiles": base_tiles,
            "pok_tiles": pok_tiles,
            "uncharted_tiles": uncharted_tiles,
            "blue_tiles": blue_tiles,
            "red_tiles": red_tiles,
            "home_tiles": home_tiles,
            "hyperlane_tiles": hyperlane_tiles
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

    print(f"Loaded {len(tiles)} tiles:")
    print(f"  - Base: {len(base_tiles)}")
    print(f"  - PoK: {len(pok_tiles)}")
    print(f"  - Blue: {len(blue_tiles)}")
    print(f"  - Red: {len(red_tiles)}")
    print(f"  - Home: {len(home_tiles)}")

    return db


def load_board_template(
    player_count: int = 6,
    template_name: str = "normal",
    project_root: Optional[Path] = None
) -> Dict:
    """
    Load a board template configuration.

    Args:
        player_count: Number of players (2-8)
        template_name: Template variant (e.g., "normal", "spiral", "large")
        project_root: Project root directory

    Returns:
        Dictionary with home_worlds, primary_tiles, secondary_tiles, etc.
    """
    if project_root is None:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "data" / "raw" / "boardData.json").exists():
                project_root = parent
                break
        if project_root is None:
            raise FileNotFoundError(
                "Could not find data/raw/boardData.json. "
                "Run scripts/update-data.sh to populate data/raw/."
            )

    board_file = project_root / "data" / "raw" / "boardData.json"

    with open(board_file, 'r') as f:
        board_data = json.load(f)

    template = board_data["styles"][str(player_count)][template_name]

    return template
