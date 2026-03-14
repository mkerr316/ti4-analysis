#!/usr/bin/env bash
# Pre-submission gate: ensure no placeholder tokens remain in docs.
# Run from project root. Exit 0 if zero matches; exit 1 otherwise.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
MATCHES=$(grep -rE 'PENDING_MAIN_EXPERIMENT|⚠ INSERT|\[INSERT\]' docs/ 2>/dev/null || true)
if [ -z "$MATCHES" ]; then
  echo "Pre-submission check: 0 placeholder matches in docs/ — OK"
  exit 0
else
  echo "Pre-submission check: placeholder matches found in docs/ — fix before submission:"
  echo "$MATCHES"
  exit 1
fi
