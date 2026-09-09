#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
# waits for the first driver to finish before touching the GPU
while pgrep -f "run_all.sh" > /dev/null 2>&1; do sleep 20; done
for cmd in \
  "python -u experiments/exp2b_saturation.py" \
  "python -u experiments/exp5_cascade.py" ; do
  echo "==================================================================="
  echo ">>> $cmd"
  echo "==================================================================="
  $cmd || echo "!!! FAILED: $cmd (continuing)"
done
echo ">>> REST DONE"
