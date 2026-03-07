"""Compare results from different optimization methods."""

import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')

# Load both results
with open(DATA_DIR / 'optimal_map_configurations_excel_method.json') as f:
    excel_configs = json.load(f)

with open(DATA_DIR / 'optimal_map_configurations_independent.json') as f:
    independent_configs = json.load(f)

print("=" * 80)
print("COMPARISON: HARD-CODED 60/40 vs INDEPENDENT BLUE/RED OPTIMIZATION")
print("=" * 80)
print()

comparison_rows = []
for pc in sorted([int(k) for k in excel_configs.keys()]):
    excel = excel_configs[str(pc)]
    independent = independent_configs[str(pc)]

    # Check if they chose the same configuration
    same_blue = excel['blue_tiles'] == independent['blue_tiles']
    same_red = excel['red_tiles'] == independent['red_tiles']
    same_config = same_blue and same_red

    comparison_rows.append({
        'Players': pc,
        'Excel Blue/Red': f"{excel['blue_tiles']}/{excel['red_tiles']}",
        'Excel SSE': round(excel['sse'], 4),
        'Independent Blue/Red': f"{independent['blue_tiles']}/{independent['red_tiles']}",
        'Independent SSE': round(independent['sse'], 4),
        'Same Config?': 'YES' if same_config else 'NO',
        'SSE Difference': round(independent['sse'] - excel['sse'], 4),
    })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))

print()
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print()

all_same = all(row['Same Config?'] == 'YES' for row in comparison_rows)

if all_same:
    print("RESULT: Both methods chose IDENTICAL blue/red splits for all player counts!")
    print()
    print("IMPLICATION: The 60/40 split is not just a game rule - it is actually")
    print("OPTIMAL for balancing planet-based vs anomaly-based objectives.")
    print()
    print("Even when given complete freedom to choose blue and red independently,")
    print("the optimizer naturally converged to the same 60/40 ratio (with rounding)")
    print("that minimizes objective difficulty variance.")
else:
    print("RESULT: Some player counts benefited from deviating from 60/40 split.")
    print()
    different_configs = [row for row in comparison_rows if row['Same Config?'] == 'NO']
    for row in different_configs:
        print(f"  {row['Players']}P: {row['Excel Blue/Red']} -> {row['Independent Blue/Red']}")
        print(f"    SSE improvement: {-row['SSE Difference']:.4f}")
        print()

print()
print("SSE Comparison:")
print(f"  Hard-coded 60/40 total SSE: {sum(excel_configs[str(pc)]['sse'] for pc in [2,3,4,5,6,7,8]):.4f}")
print(f"  Independent blue/red total SSE: {sum(independent_configs[str(pc)]['sse'] for pc in [2,3,4,5,6,7,8]):.4f}")
print()

# Split deviation analysis
print("Split Deviations from 60/40:")
for pc in sorted([int(k) for k in excel_configs.keys()]):
    config = independent_configs[str(pc)]
    blue = config['blue_tiles']
    red = config['red_tiles']
    system = blue + red

    if system > 0:
        blue_pct = (blue / system) * 100
        deviation = abs(blue_pct - 60)
        status = "Perfect" if deviation < 0.1 else f"{deviation:.1f}% deviation"

        print(f"  {pc}P: {blue}/{red} = {blue_pct:.1f}%/{100-blue_pct:.1f}% ({status})")
