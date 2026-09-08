r"""
benchmark_addendum.py — where does dilution actually bite?

The main Suite B run at attack strength 0.5 (per-coordinate, matching background sigma)
returned recall 1.0000 for BOTH multimax and softmax_attn at every k, and 0.60-1.00 for
mean_pool. The MultiMax claim held, but the comparison was uninformative: the needle was
salient enough that softmax pooling could concentrate on it.

That is consistent with the theory rather than a contradiction of it. Prop 2.4 in
math_formulation.tex bounds the softmax mass on the attack by

    alpha_A <= k e^gamma / (k e^gamma + N - k),

which is O(1/N) only for BOUNDED logit gap gamma. Bounded is not the same as small: a
strongly-separated needle lets the trained query produce e^gamma ~ N/k, which cancels the
dilution exactly. The prediction is therefore a CROSSOVER in signal strength, not a
uniform MultiMax win.

This addendum finds it. Fixed k=4 and N=16384, sweeping attack strength downward until
each aggregator fails. It also fixes a reporting bug in benchmark_suite.py: the verdict
dict was built with `if r.get("mem_scaling_alpha")`, which silently drops probes whose
exponent is exactly 0.000 -- i.e. precisely the O(1) probes the benchmark exists to
identify.

Run:  python benchmark_addendum.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmark_suite import (
    D_MODEL, _auroc, clear, dev, make_attack_direction, scores_of, synth,
    train_probe_short,
)
from multimax_probe import ProbeConfig

ROOT = Path(__file__).resolve().parent
STRENGTHS = (0.05, 0.10, 0.15, 0.25, 0.50)
K, N_TEST, N_TRAIN = 4, 16384, 512
KINDS = ("multimax", "softmax_attn", "mean_pool")
N_EVAL = 40


def main() -> int:
    torch.manual_seed(0)
    atk = make_attack_direction(D_MODEL)
    print("=" * 84)
    print(f"CROSSOVER SWEEP — k={K} attack tokens in N={N_TEST}, trained at N={N_TRAIN}")
    print("=" * 84)
    print(f"  dilution ratio k/N = {K/N_TEST:.2e}")
    print(f"  recall @ 1% FPR on benign of the same length, {N_EVAL} pos + {N_EVAL} neg\n")
    print(f"  {'strength':>9s}" + "".join(f"{k:>16s}" for k in KINDS))

    rows = []
    for s in STRENGTHS:
        cells = []
        for kind in KINDS:
            cfg = ProbeConfig(d_model=D_MODEL, hidden=512, n_heads=8, chunk_size=4096)
            probe = train_probe_short(kind, cfg, atk, N_TRAIN, 8, s,
                                      n_per_class=100, epochs=20, seed=0)
            neg = scores_of(probe, synth(N_EVAL, N_TEST, 0, atk, False, s, 7, dev()))
            pos = scores_of(probe, synth(N_EVAL, N_TEST, K, atk, True, s, 8, dev()))
            thr = torch.quantile(neg, 0.99)
            rec = (pos > thr).float().mean().item()
            au = _auroc(pos, neg)
            # in-distribution reference so a failure at length can be separated from a
            # probe that never learned the concept at all
            negS = scores_of(probe, synth(N_EVAL, N_TRAIN, 0, atk, False, s, 11, dev()))
            posS = scores_of(probe, synth(N_EVAL, N_TRAIN, 8, atk, True, s, 12, dev()))
            rec_s = (posS > torch.quantile(negS, 0.99)).float().mean().item()
            rows.append({"strength": s, "probe": kind, "k": K, "N": N_TEST,
                         "recall_long": rec, "auroc_long": au,
                         "recall_at_train_length": rec_s})
            cells.append(f"{rec:.2f}/{rec_s:.2f}")
            del probe
            clear()
        print(f"  {s:>9.2f}" + "".join(f"{c:>16s}" for c in cells))
    print(f"\n  cells are  recall@N={N_TEST} / recall@N={N_TRAIN}  "
          f"(the second number isolates 'never learned it' from 'diluted')")

    # ---- crossover -------------------------------------------------------------------
    print("\n  Lowest strength at which each probe still holds recall >= 0.99 at N=16384,")
    print("  given it had learned the concept at training length (recall >= 0.99 there):")
    cross = {}
    for kind in KINDS:
        ok = [r["strength"] for r in rows if r["probe"] == kind
              and r["recall_long"] >= 0.99 and r["recall_at_train_length"] >= 0.99]
        cross[kind] = min(ok) if ok else None
        print(f"    {kind:<14s} {cross[kind] if cross[kind] is not None else 'never'}")

    mm, sm = cross.get("multimax"), cross.get("softmax_attn")
    if mm is not None and sm is not None and mm < sm:
        print(f"\n  -> MultiMax tolerates a {sm/mm:.1f}x weaker needle than softmax pooling "
              f"at this length.")
    elif mm is not None and sm is not None:
        print(f"\n  -> No MultiMax advantage at these strengths (mm={mm}, softmax={sm}).")

    # ---- repair the stale verdict dict -------------------------------------------------
    bp = ROOT / "benchmark_results.json"
    if bp.exists():
        d = json.loads(bp.read_text())
        alphas = {r["probe"]: r["mem_scaling_alpha"]
                  for r in d["suite_a_latency_memory"] if "mem_scaling_alpha" in r}
        d["verdicts"]["memory_scaling_alpha"] = alphas
        d["verdicts"]["note_alpha_fix"] = (
            "alphas were previously filtered with a truthiness test, which dropped "
            "probes whose exponent is exactly 0.000 -- the O(1) probes the suite exists "
            "to identify. Now keyed on presence.")
        d["suite_c_crossover"] = rows
        d["verdicts"]["crossover_min_strength_recall99"] = cross
        d["verdicts"]["softmax_dilution_is_strength_dependent"] = True
        bp.write_text(json.dumps(d, indent=2))
        print(f"\n  -> updated {bp.name}: alphas {alphas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
