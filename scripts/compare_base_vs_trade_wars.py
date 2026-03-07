"""
Compare 2P/3P base game vs Trade Wars variant.

Scenarios:
1. 2P base game (universal scaling: 6 system tiles)
2. 3P base game (universal scaling: 12 system tiles)
3. 2P Trade Wars (9 blue + 8 red + 3 minor factions)
4. 3P Trade Wars (10 blue + 8 red + 3 minor factions)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')

# Load faction data for commodity values
df_factions = pd.read_csv(DATA_DIR / 'combined_faction_clusters.csv')
AVG_COMMODITIES = df_factions['commodities'].mean()  # ~3.17

print("=" * 80)
print("BASE GAME vs TRADE WARS VARIANT COMPARISON")
print("=" * 80)
print()

# Economic parameters
ROUNDS = 5

def calculate_base_game_economy(player_count):
    """Calculate economy for base game (universal scaling)."""
    # Universal scaling: pc × 5 system tiles
    system_tiles = player_count * 5
    blue_tiles = player_count * 3  # 60%
    red_tiles = player_count * 2   # 40%

    # Resources and influence (per player)
    resources = (blue_tiles * 1.5) / player_count  # ~4.5
    influence = (blue_tiles * 1.5) / player_count  # ~4.5

    # Trade goods (per player over 5 rounds)
    if player_count == 2:
        trade_goods = 3 * ROUNDS  # 15 TG (updated per user)
    elif player_count == 3:
        trade_goods = 2.5 * ROUNDS  # 12.5 TG (reluctant trade)
    else:
        # X-1 meta
        tg_per_round = 3 + (AVG_COMMODITIES * player_count)
        trade_goods = (tg_per_round / player_count) * ROUNDS

    return {
        'system_tiles': system_tiles,
        'blue_tiles': blue_tiles,
        'red_tiles': red_tiles,
        'resources': resources,
        'influence': influence,
        'trade_goods': trade_goods,
        'total_economy': resources + influence + trade_goods,
    }

def calculate_trade_wars_economy(player_count):
    """
    Calculate economy for Trade Wars variant.

    Trade Wars additions:
    - 3 Minor Faction systems (each with 2-3 planets)
    - 3 Space Stations (+1 commodity each = +3 total commodity value)
    - 3 Monuments (accumulate commodities, convert to TG)
    - All three planet traits on Minor Faction planets
    """
    if player_count == 2:
        # From variant rules
        blue_tiles = 9
        red_tiles = 8
        system_tiles = 17  # blue + red
    elif player_count == 3:
        # From variant rules
        blue_tiles = 10
        red_tiles = 8
        system_tiles = 18  # blue + red
    else:
        raise ValueError("Trade Wars only defined for 2P and 3P")

    # Minor Faction systems contribution
    # Assume each has 2.5 planets avg, ~1.5 resources and ~1.5 influence per planet
    minor_faction_planets = 3 * 2.5  # 7.5 planets total
    minor_faction_resources = minor_faction_planets * 1.5  # 11.25 total
    minor_faction_influence = minor_faction_planets * 1.5  # 11.25 total

    # Total available resources/influence (including minor factions)
    total_resources = (blue_tiles * 1.5) + minor_faction_resources
    total_influence = (blue_tiles * 1.5) + minor_faction_influence

    # Per player (assumes equal sharing of minor faction control)
    resources = total_resources / player_count
    influence = total_influence / player_count

    # Trade goods with space station boost
    # Space stations enable full Trade meta by providing washing mechanism!
    # +3 commodity value total from space stations
    boosted_avg_commodities = AVG_COMMODITIES + (3 / player_count)  # +1.5 for 2P, +1 for 3P

    # Space stations allow players to wash commodities independently
    # This makes Trade strategy card viable even at 2P/3P!
    # Use FULL X-1 meta with boosted commodity value
    tg_per_round = (3 + boosted_avg_commodities * player_count) / player_count

    # Monuments: ~1 commodity per monument per round = 3 commodities/round
    # Convert to TG over time: ~1.5 TG/round average
    monument_tg_per_round = 1.5 / player_count  # 0.75 for 2P, 0.5 for 3P

    # Total TG per round (Trade meta + monuments)
    total_tg_per_round = tg_per_round + monument_tg_per_round
    trade_goods = total_tg_per_round * ROUNDS

    return {
        'system_tiles': system_tiles,
        'blue_tiles': blue_tiles,
        'red_tiles': red_tiles,
        'minor_faction_systems': 3,
        'space_stations': 3,
        'monuments': 3,
        'resources': resources,
        'influence': influence,
        'trade_goods': trade_goods,
        'total_economy': resources + influence + trade_goods,
    }

# Calculate all four scenarios
print("SCENARIO 1: 2P BASE GAME")
print("-" * 80)
base_2p = calculate_base_game_economy(2)
print(f"  System tiles: {base_2p['system_tiles']} ({base_2p['blue_tiles']} blue + {base_2p['red_tiles']} red)")
print(f"  Resources/player: {base_2p['resources']:.2f}")
print(f"  Influence/player: {base_2p['influence']:.2f}")
print(f"  Trade goods/player: {base_2p['trade_goods']:.2f}")
print(f"  TOTAL ECONOMY/player: {base_2p['total_economy']:.2f}")
print()

print("SCENARIO 2: 3P BASE GAME")
print("-" * 80)
base_3p = calculate_base_game_economy(3)
print(f"  System tiles: {base_3p['system_tiles']} ({base_3p['blue_tiles']} blue + {base_3p['red_tiles']} red)")
print(f"  Resources/player: {base_3p['resources']:.2f}")
print(f"  Influence/player: {base_3p['influence']:.2f}")
print(f"  Trade goods/player: {base_3p['trade_goods']:.2f}")
print(f"  TOTAL ECONOMY/player: {base_3p['total_economy']:.2f}")
print()

print("SCENARIO 3: 2P TRADE WARS")
print("-" * 80)
tw_2p = calculate_trade_wars_economy(2)
print(f"  System tiles: {tw_2p['system_tiles']} ({tw_2p['blue_tiles']} blue + {tw_2p['red_tiles']} red)")
print(f"  Minor faction systems: {tw_2p['minor_faction_systems']}")
print(f"  Space stations: {tw_2p['space_stations']}")
print(f"  Monuments: {tw_2p['monuments']}")
print(f"  Resources/player: {tw_2p['resources']:.2f}")
print(f"  Influence/player: {tw_2p['influence']:.2f}")
print(f"  Trade goods/player: {tw_2p['trade_goods']:.2f}")
print(f"  TOTAL ECONOMY/player: {tw_2p['total_economy']:.2f}")
print()

print("SCENARIO 4: 3P TRADE WARS")
print("-" * 80)
tw_3p = calculate_trade_wars_economy(3)
print(f"  System tiles: {tw_3p['system_tiles']} ({tw_3p['blue_tiles']} blue + {tw_3p['red_tiles']} red)")
print(f"  Minor faction systems: {tw_3p['minor_faction_systems']}")
print(f"  Space stations: {tw_3p['space_stations']}")
print(f"  Monuments: {tw_3p['monuments']}")
print(f"  Resources/player: {tw_3p['resources']:.2f}")
print(f"  Influence/player: {tw_3p['influence']:.2f}")
print(f"  Trade goods/player: {tw_3p['trade_goods']:.2f}")
print(f"  TOTAL ECONOMY/player: {tw_3p['total_economy']:.2f}")
print()

# Comparison table
print("=" * 80)
print("COMPARISON: ECONOMIC BOOST FROM TRADE WARS")
print("=" * 80)
print()

comparison_rows = []

# 6P baseline for reference
base_6p = calculate_base_game_economy(6)

for scenario_name, config, player_count in [
    ("2P Base Game", base_2p, 2),
    ("2P Trade Wars", tw_2p, 2),
    ("3P Base Game", base_3p, 3),
    ("3P Trade Wars", tw_3p, 3),
]:
    vs_6p_pct = ((config['total_economy'] - base_6p['total_economy']) /
                 base_6p['total_economy'] * 100)

    comparison_rows.append({
        'Scenario': scenario_name,
        'Players': player_count,
        'Total Economy': round(config['total_economy'], 2),
        'vs 6P Baseline': f"{vs_6p_pct:+.1f}%",
        'Resources': round(config['resources'], 2),
        'Influence': round(config['influence'], 2),
        'Trade Goods': round(config['trade_goods'], 2),
    })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))
print()

# Trade Wars impact
print("=" * 80)
print("TRADE WARS VARIANT IMPACT")
print("=" * 80)
print()

tw_2p_boost = ((tw_2p['total_economy'] - base_2p['total_economy']) /
               base_2p['total_economy'] * 100)
tw_3p_boost = ((tw_3p['total_economy'] - base_3p['total_economy']) /
               base_3p['total_economy'] * 100)

print(f"2P Trade Wars economic boost: {tw_2p_boost:+.1f}%")
print(f"  Base: {base_2p['total_economy']:.2f} → Trade Wars: {tw_2p['total_economy']:.2f}")
print()

print(f"3P Trade Wars economic boost: {tw_3p_boost:+.1f}%")
print(f"  Base: {base_3p['total_economy']:.2f} → Trade Wars: {tw_3p['total_economy']:.2f}")
print()

# Does Trade Wars close the gap to 6P?
tw_2p_vs_6p = ((tw_2p['total_economy'] - base_6p['total_economy']) /
               base_6p['total_economy'] * 100)
tw_3p_vs_6p = ((tw_3p['total_economy'] - base_6p['total_economy']) /
               base_6p['total_economy'] * 100)

print("Does Trade Wars close the gap to 6P baseline?")
print(f"  2P Base: {((base_2p['total_economy'] - base_6p['total_economy']) / base_6p['total_economy'] * 100):+.1f}% gap")
print(f"  2P Trade Wars: {tw_2p_vs_6p:+.1f}% gap (improvement: {tw_2p_boost:.1f}%)")
print()
print(f"  3P Base: {((base_3p['total_economy'] - base_6p['total_economy']) / base_6p['total_economy'] * 100):+.1f}% gap")
print(f"  3P Trade Wars: {tw_3p_vs_6p:+.1f}% gap (improvement: {tw_3p_boost:.1f}%)")
print()

# Save results
output_data = {
    '2p_base': base_2p,
    '3p_base': base_3p,
    '2p_trade_wars': tw_2p,
    '3p_trade_wars': tw_3p,
    '6p_baseline': base_6p,
}

# Convert numpy types to native Python types for JSON serialization
def convert_to_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

output_data = convert_to_json_serializable(output_data)

output_file = DATA_DIR / 'base_vs_trade_wars_comparison.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to: {output_file}")
