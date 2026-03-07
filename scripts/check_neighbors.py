#!/usr/bin/env python3
"""
Check neighbor count for homes in 5-player vs 6-player maps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ti4_analysis.algorithms.map_generator import generate_random_map
from ti4_analysis.data.tile_loader import load_tile_database
from ti4_analysis.data.map_structures import MapSpaceType

def check_neighbor_counts():
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
        
        for i, home in enumerate(homes):
            neighbors = map_obj.get_adjacent_spaces_including_wormholes(home)
            neighbor_coords = [n.coord for n in neighbors]
            
            print(f"\nHome {i} at {home.coord}:")
            print(f"  Number of neighbors: {len(neighbors)}")
            print(f"  Neighbor coords: {neighbor_coords[:6]}")  # Show first 6
            
            # Check if any are hyperlane-connected (distance > 1)
            from ti4_analysis.algorithms.hex_grid import hex_distance
            hyperlane_neighbors = [
                n.coord for n in neighbors 
                if hex_distance(home.coord, n.coord) > 1
            ]
            if hyperlane_neighbors:
                print(f"  Hyperlane-connected: {len(hyperlane_neighbors)} neighbors")
                print(f"    {hyperlane_neighbors}")

if __name__ == "__main__":
    check_neighbor_counts()
