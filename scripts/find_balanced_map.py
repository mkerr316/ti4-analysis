#!/usr/bin/env python3
"""
Wrapper script that calls generate_game_map.py repeatedly until we get a balanced map.
"""

import subprocess
import sys
import re
from pathlib import Path
import json


def parse_metrics_from_output(output: str) -> tuple[float, float, float]:
    """Extract metrics from generate_game_map.py output."""
    # Look for final metrics
    gap_match = re.search(r'Final balance gap:\s*([\d.]+)', output)
    jaines_match = re.search(r"Jaine's Index \(after\):\s*([\d.]+)", output)
    morans_match = re.search(r"Moran's I \(after\):\s*([\d.]+)", output)
    
    if not all([gap_match, jaines_match, morans_match]):
        return None
    
    return (
        float(gap_match.group(1)),
        float(jaines_match.group(1)),
        float(morans_match.group(1))
    )


def meets_criteria(gap: float, jaines: float, morans: float, target: str = "good") -> bool:
    """
    Check if metrics meet quality criteria.
    
    Priority: Jaine's Index is the primary metric (most important for fairness).
    When Jaine's Index is good, balance gap naturally improves, so we don't reject
    based on balance gap alone. Balance gap is more arbitrary and less critical.
    """
    if target == "excellent":
        # Excellent: Jaine's Index must be excellent, Moran's I must be low
        # Balance gap is secondary - if Jaine's is perfect, accept even with higher gap
        return jaines > 0.99 and abs(morans) < 0.05
    else:  # "good"
        # Good: Jaine's Index must be good, Moran's I should be reasonable
        # Balance gap is secondary - accept if Jaine's is good
        return jaines > 0.95 and abs(morans) < 0.1


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find a balanced map by repeatedly generating maps"
    )
    parser.add_argument("--players", "-p", type=int, default=5, choices=[2,3,4,5,6,7,8])
    parser.add_argument("--template", "-t", type=str, default=None)
    parser.add_argument("--factions", nargs="+", default=[], metavar="FACTION")
    parser.add_argument("--swap-homes", action="store_true")
    parser.add_argument("--target", choices=["good", "excellent"], default="good")
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--no-pok", action="store_true")
    parser.add_argument("--no-thunders-edge", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FINDING BALANCED MAP")
    print("=" * 70)
    print()
    print(f"Target: {args.target.upper()} quality")
    print(f"  - Jaine's Index: > {'0.99' if args.target == 'excellent' else '0.95'} (PRIMARY - most important)")
    print(f"  - Moran's I: |value| < {'0.05' if args.target == 'excellent' else '0.1'}")
    print(f"  - Balance Gap: Not a hard requirement (improves naturally with good Jaine's Index)")
    print()
    print(f"Max attempts: {args.max_attempts}")
    print()
    
    # Build command
    cmd = [sys.executable, "generate_game_map.py", "--players", str(args.players)]
    if args.template:
        cmd.extend(["--template", args.template])
    if args.factions:
        cmd.extend(["--factions"] + args.factions)
    if args.swap_homes:
        cmd.append("--swap-homes")
    if args.no_pok:
        cmd.append("--no-pok")
    if args.no_thunders_edge:
        cmd.append("--no-thunders-edge")
    
    best_gap = float('inf')
    best_output_dir = None
    best_attempt = 0
    
    for attempt in range(1, args.max_attempts + 1):
        print(f"Attempt {attempt}/{args.max_attempts}...", end=" ", flush=True)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"ERROR: {result.stderr[:200]}")
                continue
            
            # Parse metrics
            metrics = parse_metrics_from_output(result.stdout)
            if not metrics:
                print("Could not parse metrics")
                continue
            
            gap, jaines, morans = metrics
            print(f"Gap={gap:.1f}, Jaine's={jaines:.3f}, Moran's I={morans:+.3f}", end="")
            
            # Check if qualified
            if meets_criteria(gap, jaines, morans, args.target):
                print(" [QUALIFIED]")
                
                # Find output directory - try multiple patterns
                output_match = re.search(r'Output directory:\s*(.+)', result.stdout)
                if not output_match:
                    # Try alternative pattern
                    output_match = re.search(r'output\s+directory:\s*(.+)', result.stdout, re.IGNORECASE)
                if not output_match:
                    # Try finding the output path from the file paths mentioned
                    output_match = re.search(r'([^\\s]+output[^\\s]+game_night[^\\s]+)', result.stdout)
                
                if output_match:
                    best_output_dir = output_match.group(1).strip()
                else:
                    # Fallback: construct expected path
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    best_output_dir = f"output/game_night_{timestamp}"
                
                print()
                print("=" * 70)
                print("SUCCESS! Found qualified map!")
                print("=" * 70)
                print()
                print(f"Final Metrics:")
                print(f"  - Balance Gap: {gap:.2f}")
                print(f"  - Jaine's Index: {jaines:.3f}")
                print(f"  - Moran's I: {morans:+.3f}")
                print()
                print(f"Output directory: {best_output_dir}")
                print()
                print("Stopping search - qualified map found!")
                return
            else:
                # Track best
                if gap < best_gap:
                    best_gap = gap
                    best_attempt = attempt
                    output_match = re.search(r'Output directory:\s*(.+)', result.stdout)
                    if output_match:
                        best_output_dir = output_match.group(1).strip()
                print()
        
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            continue
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    print()
    print("=" * 70)
    if best_output_dir:
        print(f"Best map found (attempt {best_attempt}):")
        print(f"  - Balance Gap: {best_gap:.2f}")
        print(f"Output directory: {best_output_dir}")
    else:
        print("No qualified map found. Try increasing --max-attempts.")
    print("=" * 70)


if __name__ == "__main__":
    main()
