#!/usr/bin/env bash
# Sequential driver. The experiments are run one at a time on purpose: exp1 holds ~5 GiB
# of Mistral-7B shards on a 7.96 GiB card, so anything training a probe concurrently
# would OOM rather than merely slow down.
set -u
cd "$(dirname "$0")/.."
for cmd in \
  "python -u experiments/exp1_real_residuals.py --max-n 4096" \
  "python -u experiments/exp2_ablations.py" \
  "python -u experiments/exp3_drift_calibration.py" \
  "python -u experiments/exp4_fragmentation.py" ; do
  echo "==================================================================="
  echo ">>> $cmd"
  echo "==================================================================="
  $cmd || echo "!!! FAILED: $cmd (continuing)"
done
echo ">>> ALL EXPERIMENTS DONE"
