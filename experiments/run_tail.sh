#!/usr/bin/env bash
# Strictly sequential. An earlier attempt gated with `pgrep -f run_all.sh`, which does not
# match under Git Bash on Windows, so three jobs ran at once on an 8 GiB card and all
# three stalled. Waiting on the ARTIFACT is unambiguous.
set -u
cd "$(dirname "$0")/.."
echo ">>> waiting for exp4 artifact"
for i in $(seq 1 240); do
  [ -f artifacts/exp4_fragmentation.json ] && break
  sleep 15
done
for cmd in \
  "python -u experiments/exp1_real_residuals.py --max-n 4096 --skip-extract" \
  "python -u experiments/exp5_cascade.py" ; do
  echo "==================================================================="
  echo ">>> $cmd"
  echo "==================================================================="
  $cmd || echo "!!! FAILED: $cmd (continuing)"
done
echo ">>> TAIL DONE"
