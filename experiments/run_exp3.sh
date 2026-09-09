#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
while pgrep -f "run_rest.sh" > /dev/null 2>&1; do sleep 20; done
echo ">>> exp3 rerun"
python -u experiments/exp3_drift_calibration.py || echo "!!! exp3 FAILED"
echo ">>> EXP3 DONE"
