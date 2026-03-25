#!/bin/bash
# Run TPC isolation experiment from serving/ directory
# Usage: bash experiments/tpc_isolation/run.sh [--n_requests 100] [--device cuda:0]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== TPC Isolation Experiment ==="
echo "Serving dir: $SERVING_DIR"
cd "$SERVING_DIR"

python experiments/tpc_isolation/run.py "$@"
