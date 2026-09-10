"""Fill in the residual-stream caches at N = 512 and 2048 for both model families.

exp1 (Mistral-7B-v0.1) and exp8 (Qwen2.5-7B) cached N in {256, 1024, 4096}. The bootstrap
and baseline study needs {512, 1024, 2048, 4096}, so this extracts the two missing
lengths using each experiment's own extraction routine. Nothing is reimplemented: the
prompt construction, the needle placement RNG and the layer selection all come from the
original scripts, so the new lengths are drawn from the same distribution as the cached
ones and are directly comparable to them.

Both routines skip files that already exist, so this is safe to re-run.

  python experiments/extract_extra_lengths.py [--models mistral,qwen]
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import clear                                        # noqa: E402
import experiments.exp1_real_residuals as e1                                # noqa: E402
import experiments.exp8_qwen_family as e8                                   # noqa: E402

EXTRA = [512, 2048]
N_PER_CLASS = 48


def do_mistral():
    print("== Mistral-7B-v0.1 ==")
    tok, model = e1.load_model()
    e1.collect(tok, model, EXTRA, N_PER_CLASS)
    del model, tok
    gc.collect()
    clear()


def do_qwen():
    print("== Qwen2.5-7B ==")
    tok, model = e8.load_model()
    layers = e8.layers_for(model.config.num_hidden_layers)
    print(f"  layers {layers}")
    e8.extract(tok, model, EXTRA, N_PER_CLASS, layers)
    del model, tok
    gc.collect()
    clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mistral,qwen")
    a = ap.parse_args()
    want = {m.strip() for m in a.models.split(",")}
    # Strictly one model resident at a time: two 7B models sharded onto a 7.96 GiB card
    # thrash and neither finishes.
    if "mistral" in want:
        do_mistral()
    if "qwen" in want:
        do_qwen()
    print(">>> EXTRA LENGTHS DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
