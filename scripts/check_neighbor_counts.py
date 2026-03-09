#!/usr/bin/env python3
"""
Check home neighbor counts in 5-player vs 6-player maps.
"""
import sys
from pathlib import Path

from ti4_analysis.algorithms.map_generator import generate_random_map
from ti4_analysis.data.tile_loader import load_tile_database
from ti4_analysis.data.map_structures import MapSpaceType
from ti4_analysis.algorithms.hex_grid import hex_distance

def check_neighbors():
    project_root = Path(__file__).parent.parent
    tile_db = load_tile_database(project_root=project_root)
    
    for player_count in [5, 6]:
        template = "warp" if player_count == 5 else "normal"
        print(f"\n{'='*70}")
        print(f"{player_count}-Player Map (template: {template})")
        print(f"{'='*70}")
        
        map_obj = generate_random_map(
            player_count=player_count,
            template_name=template,
            include_pok=True,
            include_thunders_edge=True,
            random_seed=42,
            tile_db=tile_db,
            project_root=project_root
        )
        
        homes = [s for s in map_obj.spaces if s.space_type == MapSpaceType.HOME]
        print(f"Number of homes: {len(homes)}")
        
        for i, home in enumerate(homes):
            neighbors = map_obj.get_adjacent_spaces_including_wormholes(home)
            print(f"\nHome {i} at {home.coord}:")
            print(f"  Total neighbors: {len(neighbors)}")
            
            # Count by distance
            distance_counts = {}
            for neighbor in neighbors:
                dist = hex_distance(home.coord, neighbor.coord)
                distance_counts[dist] = distance_counts.get(dist, 0) + 1
            
            print(f"  By distance:")
            for dist in sorted(distance_counts.keys()):
                print(f"    Distance {dist}: {distance_counts[dist]} neighbors")
                if dist > 1:
                    print(f"      ⚠️  These are hyperlane/wormhole connections!")
            
            # Show first few neighbors
            print(f"  First 6 neighbors:")
            for j, neighbor in enumerate(list(neighbors)[:6]):
                dist = hex_distance(home.coord, neighbor.coord)
                print(f"    {j+1}. {neighbor.coord} (distance={dist}, type={neighbor.space_type.value})")

if __name__ == "__main__":
    check_neighbors()
