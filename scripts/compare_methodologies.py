"""Compare map_statistics.json values with calculate_map_stats() output."""
import json
from pathlib import Path
from calculate_sse_dynamic import calculate_map_stats

DATA_DIR = Path('data')
with open(DATA_DIR / 'map_statistics.json') as f:
    ms = json.load(f)

print('Comparing map_statistics.json vs calculate_map_stats():')
print('=' * 70)
print(f"{'N':>3} {'Blue':>5} {'Red':>4} | {'JSON Res/P':>12} {'Calc Res/P':>12} | {'JSON Inf/P':>12} {'Calc Inf/P':>12}")
print('-' * 70)

for n in [2, 3, 4, 5, 6, 7, 8]:
    ms_pp = ms['by_player_count'][str(n)]['per_player']
    ms_geom = ms['by_player_count'][str(n)]['geometry']

    # Get tile counts from geometry
    blue = ms_geom['blue_systems']
    red = ms_geom['red_systems']

    # Our calculation
    ours = calculate_map_stats(blue, red, n)

    json_res = ms_pp['total_resources']
    calc_res = ours['resources_per_player']
    json_inf = ms_pp['total_influence']
    calc_inf = ours['influence_per_player']

    print(f"{n}P {blue:>5} {red:>4} | {json_res:>12.1f} {calc_res:>12.1f} | {json_inf:>12.1f} {calc_inf:>12.1f}")

print()
print("Analysis:")
print("-" * 70)
print("The JSON includes home system values (4.0 res, 4.0 inf per home planet avg)")
print("Our calc uses (3.7 res, 2.1 inf) from faction data, plus adds Mecatol (1 res, 6 inf)")
print()
print("Let's break down 6P:")
ms_6p = ms['by_player_count']['6']['per_player']
ours_6p = calculate_map_stats(18, 12, 6)
print(f"JSON 6P:")
print(f"  non_home_resources: {ms_6p['non_home_resources']:.1f}")
print(f"  home_resources: {ms_6p['home_resources']:.1f}")
print(f"  total_resources: {ms_6p['total_resources']:.1f}")
print()
print(f"Calculate 6P:")
print(f"  blue_res: {18 * 2.26:.1f}")
print(f"  red_res: {12 * 0.26:.1f}")
print(f"  home_res: {6 * 3.7:.1f}")
print(f"  mecatol_res: 1")
print(f"  total: {ours_6p['total_resources']:.1f}")
