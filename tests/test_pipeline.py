"""
Quick pipeline validation test.

This script performs a minimal N=10 experiment to verify all components work together.
Run this before executing large-scale experiments.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    from ti4_analysis.data.map_structures import (
        Planet, System, MapSpace, Evaluator, PlanetEvalStrategy
    )
    from ti4_analysis.algorithms.hex_grid import HexCoord
    from ti4_analysis.algorithms.balance_engine import TI4Map, improve_balance
    from ti4_analysis.spatial_stats.spatial_metrics import comprehensive_spatial_analysis
    from ti4_analysis.visualization.map_viz import plot_hex_map
    from ti4_analysis.evaluation.batch_experiment import run_batch_experiment
    from ti4_analysis.evaluation.analysis import analyze_experiment_results


def test_tile_loading():
    """Test tile database loading."""
    from ti4_analysis.data.tile_loader import load_tile_database, load_board_template

    db = load_tile_database(use_cache=False)
    assert len(db.tiles) > 0, f"Tile database is empty"
    assert len(db.blue_tiles) > 0, "No blue tiles found"
    assert len(db.red_tiles) > 0, "No red tiles found"

    template = load_board_template(player_count=6, template_name="normal")
    assert "home_worlds" in template, "Template missing home_worlds"
    assert "primary_tiles" in template, "Template missing primary_tiles"
    assert len(template["home_worlds"]) == 6, (
        f"Expected 6 home worlds, got {len(template['home_worlds'])}"
    )


def test_map_generation():
    """Test random map generation."""
    from ti4_analysis.algorithms.map_generator import generate_random_map

    ti4_map = generate_random_map(
        player_count=6,
        template_name="normal",
        include_pok=True,
        random_seed=42,
    )
    assert len(ti4_map.spaces) > 0, "Generated map has no spaces"
    assert len(ti4_map.get_home_spaces()) == 6, (
        f"Expected 6 home spaces, got {len(ti4_map.get_home_spaces())}"
    )
    assert len(ti4_map.get_system_spaces()) > 0, "Map has no system spaces"


def test_single_experiment():
    """Test a single experiment run."""
    from ti4_analysis.evaluation.batch_experiment import (
        run_single_experiment,
        create_joebrew_evaluator,
    )

    evaluator = create_joebrew_evaluator()
    result = run_single_experiment(
        map_id=0,
        evaluator=evaluator,
        player_count=6,
        template_name="normal",
        include_pok=True,
        optimization_iterations=50,
        random_seed=42,
        verbose=False,
    )

    assert "naive_balance_gap" in result, "Result missing naive_balance_gap"
    assert "optimized_balance_gap" in result, "Result missing optimized_balance_gap"
    assert "naive_morans_i" in result, "Result missing naive_morans_i"
    assert result["optimized_balance_gap"] <= result["naive_balance_gap"], (
        f"Optimization worsened balance gap: "
        f"{result['naive_balance_gap']:.2f} → {result['optimized_balance_gap']:.2f}"
    )


def test_mini_batch():
    """Test a mini batch experiment (N=3)."""
    from ti4_analysis.evaluation.batch_experiment import run_batch_experiment
    from ti4_analysis.evaluation.analysis import analyze_experiment_results
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        df = run_batch_experiment(
            num_maps=3,
            optimization_iterations=50,
            base_seed=42,
            output_dir=Path(tmpdir),
            verbose=False,
        )
        assert len(df) == 3, f"Expected 3 results, got {len(df)}"

        paired_results = analyze_experiment_results(df)
        assert len(paired_results) > 0, "Statistical analysis returned no results"


def main():
    """Run all validation tests as a standalone script."""
    print("=" * 80)
    print("PIPELINE VALIDATION TEST")
    print("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("Tile Loading", test_tile_loading),
        ("Map Generation", test_map_generation),
        ("Single Experiment", test_single_experiment),
        ("Mini Batch", test_mini_batch),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            test_func()
            results[test_name] = True
            print(f"✓ PASS: {test_name}")
        except Exception as e:
            import traceback
            results[test_name] = False
            print(f"✗ FAIL: {test_name} — {e}")
            traceback.print_exc()

    print("\n" + "=" * 80)
    all_passed = all(results.values())
    if all_passed:
        print("All tests passed! Pipeline is ready for full experiments.")
        return 0
    else:
        failed = [n for n, p in results.items() if not p]
        print(f"Failed: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
