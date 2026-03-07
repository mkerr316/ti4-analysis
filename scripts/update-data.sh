#!/usr/bin/env bash
# update-data.sh — sync tile data from ti4_map_generator into data/raw/
#
# Run this only when intentionally establishing a new research baseline.
# After running, review 'git diff data/raw/' before committing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
APP_ROOT="$(dirname "$REPO_ROOT")/ti4_map_generator"

if [[ ! -d "$APP_ROOT" ]]; then
    echo "ERROR: ti4_map_generator not found at $APP_ROOT" >&2
    exit 1
fi

cp "$APP_ROOT/src/data/tileData.js"                "$REPO_ROOT/data/raw/tileData.js"
cp "$APP_ROOT/src/data/boardData.json"             "$REPO_ROOT/data/raw/boardData.json"
cp "$APP_ROOT/data/processed/tiles_canonical.json" "$REPO_ROOT/data/raw/tiles_canonical.json"

echo "data/raw/ updated from $APP_ROOT"
echo "Review changes with: git diff data/raw/"
echo "Delete the tile cache to force a fresh parse: rm -f src/ti4_analysis/data/tiles_cache.json"
