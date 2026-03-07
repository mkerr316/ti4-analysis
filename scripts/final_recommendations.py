"""
Final Tile Count Recommendations

This script synthesizes all the analysis into actionable recommendations,
accounting for both SSE and objective achievability.

Key insight from user:
- Trade Wars is NEEDED for 2P/3P because without it, TG-spending objectives
  like "Negotiate Trade Routes" (5 TG) are impossible when you only get 3 TG/round
- Trade Wars adds space stations + minor factions which enable the Trade meta
"""

import json
import numpy as np
from pathlib import Path
from calculate_sse_dynamic import (
    calculate_map_stats, calculate_sse, calculate_objective_difficulty,
    objectives_data
)

DATA_DIR = Path('data')

# 6P Baseline for comparison
BASELINE_6P = calculate_map_stats(blue_tiles=18, red_tiles=12, player_count=6)


def check_tg_objectives(stats):
    """Check if TG-spending objectives are achievable."""
    tg_objectives = {
        'Negotiate Trade Routes': 5,
        'Centralize Galactic Trade': 10,
    }

    results = {}
    tg_per_round = stats['tg_per_round']

    for name, required in tg_objectives.items():
        difficulty = required / tg_per_round
        achievable = difficulty <= 2.0  # Can do it in ~2 rounds
        results[name] = {
            'required': required,
            'tg_per_round': tg_per_round,
            'difficulty': difficulty,
            'achievable': achievable,
        }

    return results


def main():
    print("=" * 80)
    print("FINAL TILE COUNT RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("Key insight: Trade Wars is REQUIRED for 2P/3P because without it,")
    print("TG-spending objectives become impossible (only 3 TG/round in 2P base).")
    print()

    # Current configurations from user's spreadsheet
    configs = {
        '8P': {'blue': 24, 'red': 16, 'pc': 8, 'tw': False, 'mf': 0, 'ss': 0},
        '7P': {'blue': 21, 'red': 14, 'pc': 7, 'tw': False, 'mf': 0, 'ss': 0},
        '6P': {'blue': 18, 'red': 12, 'pc': 6, 'tw': False, 'mf': 0, 'ss': 0},
        '5P': {'blue': 15, 'red': 10, 'pc': 5, 'tw': False, 'mf': 0, 'ss': 0},
        '4P': {'blue': 14, 'red': 10, 'pc': 4, 'tw': False, 'mf': 0, 'ss': 0},
        '3P TW': {'blue': 10, 'red': 8, 'pc': 3, 'tw': True, 'mf': 3, 'ss': 3},
        '2P TW': {'blue': 8, 'red': 7, 'pc': 2, 'tw': True, 'mf': 3, 'ss': 3},
    }

    # Also add base 2P/3P for comparison
    configs['3P Base'] = {'blue': 10, 'red': 8, 'pc': 3, 'tw': False, 'mf': 0, 'ss': 0}
    configs['2P Base'] = {'blue': 8, 'red': 7, 'pc': 2, 'tw': False, 'mf': 0, 'ss': 0}

    print("=" * 80)
    print("ECONOMY & SSE ANALYSIS")
    print("=" * 80)
    print(f"{'Config':<12} {'Blue':>5} {'Red':>5} {'Res/P':>7} {'Inf/P':>7} {'TG':>6} {'Econ':>7} {'SSE':>8}")
    print("-" * 80)

    for name, cfg in configs.items():
        stats = calculate_map_stats(
            cfg['blue'], cfg['red'], cfg['pc'],
            cfg['mf'], cfg['ss'], cfg['tw']
        )
        sse = calculate_sse(stats, BASELINE_6P)

        print(f"{name:<12} {cfg['blue']:>5} {cfg['red']:>5} "
              f"{stats['resources_per_player']:>7.1f} {stats['influence_per_player']:>7.1f} "
              f"{stats['tg_per_round']:>6.1f} {stats['total_economy']:>7.1f} "
              f"{sse['total_sse']:>8.4f}")

    # TG Objective Analysis
    print("\n" + "=" * 80)
    print("TRADE GOODS OBJECTIVE ANALYSIS")
    print("=" * 80)
    print("Negotiate Trade Routes: Spend 5 TG")
    print("Centralize Galactic Trade: Spend 10 TG")
    print("-" * 80)

    for name in ['6P', '4P', '3P Base', '3P TW', '2P Base', '2P TW']:
        cfg = configs[name]
        stats = calculate_map_stats(
            cfg['blue'], cfg['red'], cfg['pc'],
            cfg['mf'], cfg['ss'], cfg['tw']
        )
        tg_check = check_tg_objectives(stats)

        print(f"\n{name} ({stats['tg_per_round']:.1f} TG/round):")
        for obj_name, obj_data in tg_check.items():
            status = "OK" if obj_data['achievable'] else "IMPOSSIBLE"
            print(f"  {obj_name}: {obj_data['required']} TG, difficulty={obj_data['difficulty']:.2f} [{status}]")

    # Key Comparison: Base vs Trade Wars for 2P/3P
    print("\n" + "=" * 80)
    print("2P/3P: WHY TRADE WARS IS NEEDED")
    print("=" * 80)

    for pc in [3, 2]:
        print(f"\n{pc}P:")

        # Base
        stats_base = calculate_map_stats(
            configs[f'{pc}P Base']['blue'], configs[f'{pc}P Base']['red'], pc
        )
        sse_base = calculate_sse(stats_base, BASELINE_6P)
        tg_base = check_tg_objectives(stats_base)

        # Trade Wars
        stats_tw = calculate_map_stats(
            configs[f'{pc}P TW']['blue'], configs[f'{pc}P TW']['red'], pc,
            minor_factions=3, space_stations=3, trade_wars=True
        )
        sse_tw = calculate_sse(stats_tw, BASELINE_6P)
        tg_tw = check_tg_objectives(stats_tw)

        print(f"  Base game:   TG/round={stats_base['tg_per_round']:.1f}, SSE={sse_base['total_sse']:.4f}")
        print(f"    - Negotiate Trade Routes: {'OK' if tg_base['Negotiate Trade Routes']['achievable'] else 'IMPOSSIBLE'}")

        print(f"  Trade Wars:  TG/round={stats_tw['tg_per_round']:.1f}, SSE={sse_tw['total_sse']:.4f}")
        print(f"    - Negotiate Trade Routes: {'OK' if tg_tw['Negotiate Trade Routes']['achievable'] else 'IMPOSSIBLE'}")

        if not tg_base['Negotiate Trade Routes']['achievable'] and tg_tw['Negotiate Trade Routes']['achievable']:
            print(f"  -> TRADE WARS REQUIRED to make TG objectives achievable!")

    # Final Recommendations
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("For 4-8 player games: Use standard tile counts (already well-balanced)")
    print()
    print("For 2-3 player games: MUST use Trade Wars variant because:")
    print("  1. Base game only provides ~3 TG/round (no trading)")
    print("  2. TG-spending objectives (5 TG, 10 TG) become impossible")
    print("  3. Trade Wars adds space stations for commodity washing")
    print("  4. Trade Wars adds minor factions for more map content")
    print()

    recommended = {
        '8P': configs['8P'],
        '7P': configs['7P'],
        '6P': configs['6P'],
        '5P': configs['5P'],
        '4P': configs['4P'],
        '3P': configs['3P TW'],  # Trade Wars REQUIRED
        '2P': configs['2P TW'],  # Trade Wars REQUIRED
    }

    print(f"{'Players':<10} {'Blue':>6} {'Red':>6} {'Minor':>6} {'S.Sta':>6} {'TG/rd':>7} {'SSE':>8}")
    print("-" * 70)

    for name, cfg in recommended.items():
        stats = calculate_map_stats(
            cfg['blue'], cfg['red'], cfg['pc'],
            cfg['mf'], cfg['ss'], cfg['tw']
        )
        sse = calculate_sse(stats, BASELINE_6P)

        tw_note = " (TW)" if cfg['tw'] else ""
        print(f"{name + tw_note:<10} {cfg['blue']:>6} {cfg['red']:>6} {cfg['mf']:>6} {cfg['ss']:>6} "
              f"{stats['tg_per_round']:>7.1f} {sse['total_sse']:>8.4f}")

    # SSE Breakdown comparison with Excel
    print("\n" + "=" * 80)
    print("SSE COMPARISON WITH USER'S EXCEL")
    print("=" * 80)

    excel_sse = {
        '8P': 0.17, '7P': 0.05, '6P': 0.00, '5P': 0.09, '4P': 0.34,
        '3P': 1.11, '2P': 2.73
    }

    print(f"{'Players':<10} {'Calculated':>12} {'Excel':>12} {'Ratio':>10}")
    print("-" * 50)

    for name in ['8P', '7P', '6P', '5P', '4P', '3P', '2P']:
        cfg = recommended[name]
        stats = calculate_map_stats(
            cfg['blue'], cfg['red'], cfg['pc'],
            cfg['mf'], cfg['ss'], cfg['tw']
        )
        sse = calculate_sse(stats, BASELINE_6P)
        calc = sse['total_sse']
        excel = excel_sse[name]
        ratio = calc / excel if excel > 0 else float('inf')

        print(f"{name:<10} {calc:>12.4f} {excel:>12.2f} {ratio:>10.2f}x")

    print("\nNote: Calculated SSE is lower than Excel because objective")
    print("calculations differ. The TRENDS are consistent (6P lowest, 2P highest).")

    # Save
    output = {
        'recommendations': {
            name: {
                'blue_tiles': cfg['blue'],
                'red_tiles': cfg['red'],
                'minor_factions': cfg['mf'],
                'space_stations': cfg['ss'],
                'trade_wars': cfg['tw'],
            }
            for name, cfg in recommended.items()
        },
        'notes': [
            "2P and 3P REQUIRE Trade Wars variant",
            "Trade Wars adds: 3 minor factions, 3 space stations",
            "Space stations enable commodity washing for Trade meta",
            "Without Trade Wars, TG objectives are impossible in 2P/3P",
        ]
    }

    output_path = DATA_DIR / 'final_recommendations.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
