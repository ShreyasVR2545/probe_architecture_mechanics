#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
while pgrep -f "run_exp3.sh|run_rest.sh|run_all.sh" > /dev/null 2>&1; do sleep 20; done
echo ">>> exp1 rerun: needle-disjoint vs index split (cached residuals, no extraction)"
python -u experiments/exp1_real_residuals.py --max-n 4096 --skip-extract || echo "!!! FAILED"
echo ">>> EXP1B DONE"
