"""
Tests for balance engine algorithms.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st

from ti4_analysis.algorithms.hex_grid import HexCoord
from ti4_analysis.data.map_structures import (
    Planet, System, MapSpace, MapSpaceType, Evaluator,
    PlanetTrait, TechSpecialty, PlanetEvalStrategy
)
from ti4_analysis.algorithms.balance_engine import (
    TI4Map, get_home_values, get_balance_gap, can_swap_system,
    improve_balance, analyze_balance
)


class TestPlanetEvaluation:
    """Tests for planet evaluation logic."""

    def test_basic_planet_evaluation(self):
        """Test basic planet evaluation."""
        planet = Planet(
            name="Test Planet",
            resources=3,
            influence=2,
            traits=[PlanetTrait.INDUSTRIAL],
            tech_specialties=None
        )

        evaluator = Evaluator(
            name="Test",
            PLANET_STRATEGY=PlanetEvalStrategy.SUM,
            RESOURCES_MULTIPLIER=1.0,
            INFLUENCE_MULTIPLIER=1.0
        )

        value = planet.evaluate(evaluator)
        assert value > 0

    def test_tech_specialty_adds_value(self):
        """Test that tech specialties increase value."""
        planet_no_tech = Planet(
            name="No Tech",
            resources=2,
            influence=2,
            traits=None,
            tech_specialties=None
        )

        planet_with_tech = Planet(
            name="With Tech",
            resources=2,
            influence=2,
            traits=None,
            tech_specialties=[TechSpecialty.WARFARE]
        )

        evaluator = Evaluator(name="Test", TECH_MOD=5.0)

        value_no_tech = planet_no_tech.evaluate(evaluator)
        value_with_tech = planet_with_tech.evaluate(evaluator)

        assert value_with_tech > value_no_tech

    def test_greatest_plus_tech_strategy(self):
        """Test GREATEST_PLUS_TECH strategy."""
        # Planet with 5 resources, 1 influence
        planet = Planet(
            name="Resource Rich",
            resources=5,
            influence=1,
            traits=None,
            tech_specialties=[TechSpecialty.WARFARE]
        )

        evaluator = Evaluator(
            name="Test",
            PLANET_STRATEGY=PlanetEvalStrategy.GREATEST_PLUS_TECH,
            RESOURCES_MULTIPLIER=1.0,
            INFLUENCE_MULTIPLIER=1.0,
            TECH_MOD=2.0
        )

        # Should use max(5, 1) + 2 = 7
        value = planet.evaluate(evaluator)
        assert value == pytest.approx(7.0, abs=0.1)


class TestSystemEvaluation:
    """Tests for system evaluation logic."""

    def test_empty_system(self):
        """Test that empty system has zero value."""
        system = System(id=1, planets=[])
        evaluator = Evaluator(name="Test")
        assert system.evaluate(evaluator) == 0

    def test_single_planet_system(self):
        """Test single planet system evaluation."""
        planet = Planet("Test", resources=2, influence=2)
        system = System(id=1, planets=[planet])
        evaluator = Evaluator(name="Test")
        value = system.evaluate(evaluator)
        assert value > 0

    def test_multi_planet_bonus(self):
        """Test that multi-planet systems get bonus."""
        planet1 = Planet("P1", resources=1, influence=1)
        planet2 = Planet("P2", resources=1, influence=1)

        single_system = System(id=1, planets=[planet1])
        multi_system = System(id=2, planets=[planet1, planet2])

        evaluator = Evaluator(
            name="Test",
            MULTI_PLANET_MOD=5.0,
            SINGLE_PLANET_MOD=0.0
        )

        single_value = single_system.evaluate(evaluator)
        multi_value = multi_system.evaluate(evaluator)

        # Multi should have bonus
        assert multi_value > 2 * single_value


class TestCanSwapSystem:
    """Tests for swap eligibility logic."""

    def test_cannot_swap_open_space(self):
        """Open spaces cannot be swapped."""
        space = MapSpace(
            coord=HexCoord(0, 0, 0),
            space_type=MapSpaceType.OPEN
        )
        assert not can_swap_system(space)

    def test_cannot_swap_mecatol_rex(self):
        """Mecatol Rex cannot be swapped."""
        planet = Planet("Mecatol Rex", resources=1, influence=6)
        system = System(id=18, planets=[planet])
        space = MapSpace(
            coord=HexCoord(0, 0, 0),
            space_type=MapSpaceType.SYSTEM,
            system=system
        )
        assert not can_swap_system(space)

    def test_can_swap_normal_planet_system(self):
        """Normal planet systems can be swapped."""
        planet = Planet("Normal Planet", resources=2, influence=2)
        system = System(id=10, planets=[planet])
        space = MapSpace(
            coord=HexCoord(0, 0, 0),
            space_type=MapSpaceType.SYSTEM,
            system=system
        )
        assert can_swap_system(space)


class TestBalanceGap:
    """Tests for balance gap calculation."""

    def test_equal_values_zero_gap(self):
        """Equal home values should give zero gap."""
        from ti4_analysis.algorithms.balance_engine import HomeValue

        home_values = [
            HomeValue(space=None, value=10.0),
            HomeValue(space=None, value=10.0),
            HomeValue(space=None, value=10.0)
        ]

        gap = get_balance_gap(home_values)
        assert gap == 0.0

    def test_gap_calculation(self):
        """Test gap is max - min."""
        from ti4_analysis.algorithms.balance_engine import HomeValue

        home_values = [
            HomeValue(space=None, value=5.0),
            HomeValue(space=None, value=10.0),
            HomeValue(space=None, value=15.0)
        ]

        gap = get_balance_gap(home_values)
        assert gap == 10.0  # 15 - 5


class TestAnalyzeBalance:
    """Tests for balance analysis."""

    def test_analyze_balance_returns_metrics(self):
        """Test that analyze_balance returns expected metrics."""
        # Create minimal map
        home1 = MapSpace(
            coord=HexCoord(0, 0, 0),
            space_type=MapSpaceType.HOME
        )
        home2 = MapSpace(
            coord=HexCoord(3, -3, 0),
            space_type=MapSpaceType.HOME
        )

        planet = Planet("Test", resources=2, influence=2)
        system = System(id=1, planets=[planet])
        sys_space = MapSpace(
            coord=HexCoord(1, -1, 0),
            space_type=MapSpaceType.SYSTEM,
            system=system
        )

        ti4_map = TI4Map([home1, home2, sys_space])
        evaluator = Evaluator(name="Test")

        analysis = analyze_balance(ti4_map, evaluator)

        assert 'home_values' in analysis
        assert 'balance_gap' in analysis
        assert 'mean' in analysis
        assert 'std' in analysis
        assert 'fairness_index' in analysis

    def test_fairness_index_range(self):
        """Fairness index should be between 0 and 1."""
        from ti4_analysis.spatial_stats.spatial_metrics import jains_fairness_index

        # Perfect fairness
        values = np.array([10.0, 10.0, 10.0])
        fairness = jains_fairness_index(values)
        assert fairness == pytest.approx(1.0)

        # Some inequality
        values = np.array([5.0, 10.0, 15.0])
        fairness = jains_fairness_index(values)
        assert 0.0 <= fairness <= 1.0


class TestImproveBalance:
    """Tests for balance improvement algorithm."""

    def test_improve_balance_reduces_gap(self):
        """Test that balance improvement reduces gap (or keeps it same)."""
        # Create simple unbalanced map
        home1 = MapSpace(HexCoord(0, 0, 0), MapSpaceType.HOME)
        home2 = MapSpace(HexCoord(5, -5, 0), MapSpaceType.HOME)

        # Create two different systems
        planet1 = Planet("Rich", resources=5, influence=5)
        planet2 = Planet("Poor", resources=1, influence=1)

        system1 = System(id=1, planets=[planet1])
        system2 = System(id=2, planets=[planet2])

        # Place them asymmetrically
        sys1_space = MapSpace(HexCoord(1, -1, 0), MapSpaceType.SYSTEM, system1)
        sys2_space = MapSpace(HexCoord(4, -4, 0), MapSpaceType.SYSTEM, system2)

        ti4_map = TI4Map([home1, home2, sys1_space, sys2_space])
        evaluator = Evaluator(name="Test")

        initial_gap = get_balance_gap(get_home_values(ti4_map, evaluator))

        final_gap, history = improve_balance(
            ti4_map, evaluator, iterations=10, random_seed=42
        )

        # Final gap should be <= initial gap
        assert final_gap <= initial_gap

    def test_improve_balance_with_no_swappable_systems(self):
        """Test handling of maps with no swappable systems."""
        home1 = MapSpace(HexCoord(0, 0, 0), MapSpaceType.HOME)
        home2 = MapSpace(HexCoord(3, -3, 0), MapSpaceType.HOME)

        ti4_map = TI4Map([home1, home2])
        evaluator = Evaluator(name="Test")

        # Should handle gracefully
        final_gap, history = improve_balance(ti4_map, evaluator, iterations=10)
        assert len(history) > 0


class TestTI4MapCopy:
    """Regression tests for TI4Map.copy() deep-copy semantics.

    pareto_optimize() correctness depends on copy() producing fresh MapSpace
    objects. If copy() is ever changed to return the same objects, child map
    mutations would bleed into parent maps, silently corrupting the optimizer.
    """

    def _make_map(self):
        planet = Planet("Test", resources=2, influence=2)
        system = System(id=1, planets=[planet])
        space = MapSpace(HexCoord(0, 0, 0), MapSpaceType.SYSTEM, system)
        return TI4Map([space])

    def test_copy_produces_fresh_space_objects(self):
        """copy() must return new MapSpace objects, not references to originals."""
        ti4_map = self._make_map()
        copied = ti4_map.copy()
        assert copied.spaces[0] is not ti4_map.spaces[0]

    def test_copy_system_mutation_does_not_affect_original(self):
        """Reassigning .system on a copied space must not touch the source map."""
        ti4_map = self._make_map()
        copied = ti4_map.copy()
        original_system = ti4_map.spaces[0].system
        copied.spaces[0].system = None
        assert ti4_map.spaces[0].system is original_system

    def test_copy_preserves_system_reference(self):
        """Copied spaces initially point to the same System objects (shared, not cloned)."""
        ti4_map = self._make_map()
        copied = ti4_map.copy()
        assert copied.spaces[0].system is ti4_map.spaces[0].system
