"""
TI4 map data structures: Planets, Systems, and Maps.

Ported from JavaScript implementation in src/balance/map-logic.js
"""

from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy

from ..algorithms.hex_grid import HexCoord, hex_distance, get_adjacent_coordinates


class PlanetTrait(Enum):
    """Planet trait types."""
    CULTURAL = "cultural"
    HAZARDOUS = "hazardous"
    INDUSTRIAL = "industrial"


class TechSpecialty(Enum):
    """Technology specialty types."""
    WARFARE = "warfare"         # Red
    PROPULSION = "propulsion"   # Blue
    BIOTIC = "biotic"           # Green
    CYBERNETIC = "cybernetic"   # Yellow


class Anomaly(Enum):
    """Anomaly types."""
    ASTEROID_FIELD = "asteroid_field"
    GRAVITY_RIFT = "gravity_rift"
    NEBULA = "nebula"
    SUPERNOVA = "supernova"
    ENTROPIC_SCAR = "entropic_scar"


class Wormhole(Enum):
    """Wormhole types."""
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    DELTA = "delta"
    EPSILON = "epsilon"
    ZETA = "zeta"
    ETA = "eta"
    THETA = "theta"
    IOTA = "iota"
    KAPPA = "kappa"


class MapSpaceType(Enum):
    """Map space types."""
    OPEN = "open"
    SYSTEM = "system"
    HOME = "home"
    CLOSED = "closed"
    WARP = "warp"


class PlanetEvalStrategy(Enum):
    """
    Planet evaluation strategies.

    SUM: Add all values (R + I + T)
    GREATEST: Take max value max(R, I, T)
    GREATEST_PLUS_TECH: max(R, I) + T (Milty Draft style)
    """
    SUM = "1"
    GREATEST = "2"
    GREATEST_PLUS_TECH = "3"


@dataclass
class Planet:
    """Represents a planet in TI4."""

    name: str
    resources: int
    influence: int
    traits: Optional[List[PlanetTrait]] = None
    tech_specialties: Optional[List[TechSpecialty]] = None

    def evaluate(self, evaluator: 'Evaluator') -> float:
        """
        Evaluate planet value using given evaluator parameters.

        Args:
            evaluator: Evaluator with multipliers and modifiers

        Returns:
            Computed planet value
        """
        value = evaluator.BASE_PLANET_MOD

        # Calculate base values
        r = self.resources * evaluator.RESOURCES_MULTIPLIER
        i = self.influence * evaluator.INFLUENCE_MULTIPLIER
        t = 0.0

        # Tech specialty value
        if self.tech_specialties:
            t += evaluator.TECH_MOD
            for tech in self.tech_specialties:
                if tech == TechSpecialty.WARFARE:
                    t += evaluator.TECH_WARFARE_MOD
                elif tech == TechSpecialty.PROPULSION:
                    t += evaluator.TECH_PROPULSION_MOD
                elif tech == TechSpecialty.BIOTIC:
                    t += evaluator.TECH_BIOTIC_MOD
                elif tech == TechSpecialty.CYBERNETIC:
                    t += evaluator.TECH_CYBERNETIC_MOD

        # Apply strategy
        if evaluator.PLANET_STRATEGY == PlanetEvalStrategy.GREATEST:
            value += max(r, i, t)
        elif evaluator.PLANET_STRATEGY == PlanetEvalStrategy.GREATEST_PLUS_TECH:
            value += max(r, i) + t
        else:  # SUM
            value += r + i + t

        # Nonzero bonuses
        if r > 0:
            value += evaluator.NONZERO_RESOURCES_MOD
        if i > 0:
            value += evaluator.NONZERO_INFLUENCE_MOD

        # Trait bonuses
        if self.traits:
            for trait in self.traits:
                if trait == PlanetTrait.CULTURAL:
                    value += evaluator.TRAIT_CULTURAL_MOD
                elif trait == PlanetTrait.HAZARDOUS:
                    value += evaluator.TRAIT_HAZARDOUS_MOD
                elif trait == PlanetTrait.INDUSTRIAL:
                    value += evaluator.TRAIT_INDUSTRIAL_MOD

        return value

    def __repr__(self):
        tech_str = ""
        if self.tech_specialties:
            tech_map = {
                TechSpecialty.WARFARE: "R",
                TechSpecialty.PROPULSION: "B",
                TechSpecialty.BIOTIC: "G",
                TechSpecialty.CYBERNETIC: "Y"
            }
            tech_str = "".join(tech_map[t] for t in self.tech_specialties)
            tech_str = f"/{tech_str}"
        return f"{self.name} ({self.resources}/{self.influence}{tech_str})"


@dataclass
class System:
    """Represents a system tile in TI4."""

    id: int
    planets: List[Planet] = field(default_factory=list)
    anomalies: Optional[List[Anomaly]] = None
    wormhole: Optional[Wormhole] = None

    def evaluate(self, evaluator: 'Evaluator') -> float:
        """
        Evaluate system value using given evaluator parameters.

        Args:
            evaluator: Evaluator with multipliers and modifiers

        Returns:
            Computed system value
        """
        value = 0.0

        # Sum planet values
        for planet in self.planets:
            value += planet.evaluate(evaluator)

        # Multi-planet bonuses
        if len(self.planets) == 1:
            value += evaluator.SINGLE_PLANET_MOD
        elif len(self.planets) == 2:
            value += evaluator.MULTI_PLANET_MOD
            # Check for matching traits
            if (self.planets[0].traits and self.planets[1].traits and
                any(t in self.planets[1].traits for t in self.planets[0].traits)):
                value += evaluator.MATCHING_PLANETS_MOD
            else:
                value += evaluator.NONMATCHING_PLANETS_MOD
        elif len(self.planets) > 2:
            value += 2 * evaluator.MULTI_PLANET_MOD
            value += evaluator.MATCHING_PLANETS_MOD

        # Special system bonuses
        if self.is_mecatol_rex():
            value += evaluator.MECATOL_REX_SYS_MOD
        if self.is_legendary():
            value += evaluator.LEGENDARY_PLANET_SYS_MOD
        if self.is_space_station():
            value += evaluator.SPACE_STATION_SYS_MOD

        return value

    def is_mecatol_rex(self) -> bool:
        """Check if this is the Mecatol Rex system."""
        return any(p.name == "Mecatol Rex" for p in self.planets)

    def is_legendary(self) -> bool:
        """Check if this system contains a legendary planet."""
        # Legendary system IDs from map-logic.js:154
        return self.id in [18, 65, 66, 97, 98, 99, 100, 115]

    def is_space_station(self) -> bool:
        """Check if this system contains a space station."""
        # Space station system IDs from map-logic.js:157
        return self.id in [109, 111, 117]

    def is_red(self) -> bool:
        """Check if this is a red (hazardous) system."""
        return len(self.planets) == 0 or self.anomalies is not None

    def is_blue(self) -> bool:
        """Check if this is a blue (normal) system."""
        return not self.is_red() and not self.is_mecatol_rex()

    def get_distance_modifier(
        self,
        evaluator: 'Evaluator',
        through_wormhole: bool = False
    ) -> Optional[float]:
        """
        Calculate distance modifier for this system.

        Returns None if system blocks pathing (e.g., supernova).

        Args:
            evaluator: Evaluator with distance modifiers
            through_wormhole: Whether accessed via wormhole

        Returns:
            Distance modifier value, or None if impassable
        """
        value = evaluator.DISTANCE_MOD_BASE

        # Blue systems or Mecatol Rex
        if self.is_blue() or self.is_mecatol_rex():
            if evaluator.DISTANCE_MOD_PLANET is False:
                return None
            value += evaluator.DISTANCE_MOD_PLANET
            if self.wormhole and through_wormhole:
                if evaluator.DISTANCE_MOD_PLANET_WORMHOLE is False:
                    return None
                value += evaluator.DISTANCE_MOD_PLANET_WORMHOLE

        # Empty space
        elif self.anomalies is None:
            if self.wormhole and through_wormhole:
                if evaluator.DISTANCE_MOD_EMPTY_WORMHOLE is False:
                    return None
                value += evaluator.DISTANCE_MOD_EMPTY_WORMHOLE
            else:
                if evaluator.DISTANCE_MOD_EMPTY is False:
                    return None
                value += evaluator.DISTANCE_MOD_EMPTY

        # Anomaly modifiers
        if self.anomalies:
            for anomaly in self.anomalies:
                mod_value = {
                    Anomaly.ASTEROID_FIELD: evaluator.DISTANCE_MOD_ASTEROID_FIELD,
                    Anomaly.GRAVITY_RIFT: evaluator.DISTANCE_MOD_GRAVITY_RIFT,
                    Anomaly.NEBULA: evaluator.DISTANCE_MOD_NEBULA,
                    Anomaly.SUPERNOVA: evaluator.DISTANCE_MOD_SUPERNOVA,
                    Anomaly.ENTROPIC_SCAR: evaluator.DISTANCE_MOD_ENTROPIC_SCAR,
                }.get(anomaly)

                if mod_value is False:
                    return None
                value += mod_value or 0

        return value

    def __repr__(self):
        parts = [str(p) for p in self.planets]
        if not parts:
            parts.append("Empty Space")
        if self.wormhole:
            parts.append(f"({self.wormhole.value[0]})")
        if self.anomalies:
            parts.extend([a.value.replace('_', ' ').title() for a in self.anomalies])
        return f"System {self.id}: {', '.join(parts)}"


@dataclass
class Evaluator:
    """
    Parameter set for evaluating planet and system values.

    This represents one of the evaluator strategies (Simple Slice, Joebrew, etc.)
    """

    name: str
    PLANET_STRATEGY: PlanetEvalStrategy = PlanetEvalStrategy.GREATEST_PLUS_TECH

    # Base multipliers
    RESOURCES_MULTIPLIER: float = 3.0
    INFLUENCE_MULTIPLIER: float = 2.0
    TECH_MOD: float = 5.0

    # Tech specialty bonuses
    TECH_WARFARE_MOD: float = 0.0
    TECH_PROPULSION_MOD: float = 0.0
    TECH_BIOTIC_MOD: float = 0.0
    TECH_CYBERNETIC_MOD: float = 0.0

    # Planet modifiers
    BASE_PLANET_MOD: float = 0.0
    SINGLE_PLANET_MOD: float = 0.0
    MULTI_PLANET_MOD: float = 1.0
    MATCHING_PLANETS_MOD: float = 1.0
    NONMATCHING_PLANETS_MOD: float = 0.0

    # Nonzero bonuses
    NONZERO_RESOURCES_MOD: float = 0.0
    NONZERO_INFLUENCE_MOD: float = 0.0

    # Trait modifiers
    TRAIT_CULTURAL_MOD: float = 0.0
    TRAIT_HAZARDOUS_MOD: float = 0.0
    TRAIT_INDUSTRIAL_MOD: float = 0.0

    # Special systems
    MECATOL_REX_SYS_MOD: float = 6.0
    LEGENDARY_PLANET_SYS_MOD: float = 6.0
    SPACE_STATION_SYS_MOD: float = 5.0

    # Distance modifiers
    DISTANCE_MULTIPLIER: List[float] = field(default_factory=lambda: [6, 6, 6, 4, 4, 2, 1])
    DISTANCE_MOD_BASE: float = 0.0
    DISTANCE_MOD_PLANET: float = 0.0
    DISTANCE_MOD_PLANET_WORMHOLE: float = 0.0
    DISTANCE_MOD_EMPTY: float = 0.0
    DISTANCE_MOD_EMPTY_WORMHOLE: float = 0.0
    DISTANCE_MOD_ASTEROID_FIELD: float = False  # False = impassable (Round 0); see docs/limitations/anomalies.md
    DISTANCE_MOD_GRAVITY_RIFT: float = 0.0
    DISTANCE_MOD_NEBULA: float = 0.0  # 0 = logistically transparent for Move-1; see docs/limitations/anomalies.md
    DISTANCE_MOD_SUPERNOVA: float = False  # Blocks path
    DISTANCE_MOD_ENTROPIC_SCAR: float = 0.0
    DISTANCE_MOD_ADJACENT_TO_OPPONENT: float = 0.0

    def get_distance_multiplier(self, modded_distance: float) -> float:
        """
        Get distance multiplier for a given (modified) distance.

        Args:
            modded_distance: Modified distance value

        Returns:
            Multiplier for this distance
        """
        d = int(max(0, min(modded_distance, 10)))
        if d < len(self.DISTANCE_MULTIPLIER):
            return self.DISTANCE_MULTIPLIER[d]
        return 0.0


@dataclass
class MapSpace:
    """Represents a single hex space on the map."""

    coord: HexCoord
    space_type: MapSpaceType = MapSpaceType.OPEN
    system: Optional[System] = None

    def __hash__(self):
        return hash(self.coord)

    def __eq__(self, other):
        if not isinstance(other, MapSpace):
            return False
        return self.coord == other.coord
