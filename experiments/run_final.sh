#!/usr/bin/env bash
# One script, plain sequence, no cross-process gating. An earlier attempt used
# `pgrep -f` to wait on a sibling script; that never matches under Git Bash on Windows,
# so three GPU jobs ran at once on an 8 GiB card and all three stalled.
set -u
cd "$(dirname "$0")/.."

run () {
  echo "==================================================================="
  echo ">>> $*"
  echo "==================================================================="
  "$@" || echo "!!! FAILED: $* (continuing)"
}

# exp6 and exp7 need no new weights, so they run while Qwen is still downloading.
run python -u experiments/exp6_pareto.py
run python -u experiments/exp7_cascade_scaled.py

echo ">>> waiting for the Qwen download before exp8"
for i in $(seq 1 400); do
  grep -q DOWNLOADED artifacts/_qwen_download.log 2>/dev/null && break
  sleep 20
done
if grep -q DOWNLOADED artifacts/_qwen_download.log 2>/dev/null; then
  echo ">>> Qwen present"
  run python -u experiments/exp8_qwen_family.py --max-n 4096
else
  echo "!!! Qwen download never completed; exp8 skipped"
fi
echo ">>> FINAL RUNS DONE"
