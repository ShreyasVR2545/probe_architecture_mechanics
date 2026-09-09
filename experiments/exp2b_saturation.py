"""Experiment 2b: is the c x H failure really clamp saturation at deployment length?

The grid in exp2 shows cells that train to a LOW loss at N=512 and then score exactly
0.500 AUROC at N=16,384. Exactly chance, with ties everywhere, is the signature of a
constant output, and the natural suspect is the bounded link: if the pre-clamp logit
exceeds c for every sequence at evaluation length, every score collapses onto the same
value and all ranking information is destroyed.

That is a hypothesis, not a result, so this measures it directly. For each cell we train
as exp2 does and then record the PRE-clamp logit (via raw_logit) at the training length
and at the evaluation length, together with the fraction of evaluation sequences whose
raw logit has already left the interval [-c, c].

The prediction, if the hypothesis is right: cells with AUROC 0.500 should show a saturated
fraction near 1.0 at N=16,384 while remaining unsaturated at N=512, and the raw logit
should scale with H, since the logit is a SUM of H per-head maxima and Remark 2.6 says
each grows like tau*sqrt(2 log N).

  python experiments/exp2b_saturation.py [--quick]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimax_probe import ProbeConfig                                       # noqa: E402
from experiments.common import (auroc, clear, dev, make_attack_direction,     # noqa: E402
                                save, scores_iter, synth_iter, train_probe)

D_MODEL, N_TRAIN, N_EVAL, STRENGTH, K = 2048, 512, 16_384, 0.15, 8


@torch.no_grad()
def raw_stats(probe, it, c):
    """Mean |raw logit| and the fraction of sequences already outside [-c, c]."""
    vals = []
    for x in it:
        xg = x.to(dev())
        vals.append(float(probe.raw_logit(xg).reshape(-1)[0]))
        del xg, x
    t = torch.tensor(vals)
    return {"mean_raw": float(t.mean()), "mean_abs_raw": float(t.abs().mean()),
            "max_abs_raw": float(t.abs().max()),
            "saturated_frac": float((t.abs() > c).float().mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    atk = make_attack_direction(D_MODEL)
    cells = [(1.0, 8), (2.0, 8), (5.0, 1), (5.0, 8), (10.0, 1), (10.0, 8), (10.0, 32)]
    if a.quick:
        cells = [(2.0, 8), (5.0, 8), (10.0, 1)]
    n_eval = 20 if a.quick else 32
    rows = []
    for c, H in cells:
        cfg = ProbeConfig(d_model=D_MODEL, hidden=512, n_heads=H, chunk_size=4096,
                          logit_clamp=c, straight_through_clamp=True)
        probe = train_probe("multimax", cfg, atk, N_train=N_TRAIN, k_train=K,
                            strength=STRENGTH, n_per_class=32 if a.quick else 96,
                            epochs=6 if a.quick else 12, seed=7)
        st_tr = raw_stats(probe, synth_iter(n_eval, N_TRAIN, K, atk, True, STRENGTH, 31), c)
        st_ev = raw_stats(probe, synth_iter(n_eval, N_EVAL, K, atk, True, STRENGTH, 998), c)
        p = scores_iter(probe, synth_iter(n_eval, N_EVAL, K, atk, True, STRENGTH, 998))
        q = scores_iter(probe, synth_iter(n_eval, N_EVAL, 0, atk, False, STRENGTH, 999))
        au = auroc(p, q)
        rows.append({"c": c, "H": H, "auroc_ood": au,
                     "train_len": {"N": N_TRAIN, **st_tr},
                     "eval_len": {"N": N_EVAL, **st_ev},
                     "predicted_scale_H_sqrt2logN":
                         H * math.sqrt(2 * math.log(N_EVAL))})
        print(f"  c={c:<5} H={H:<3} AUROC={au:.3f} | "
              f"N={N_TRAIN}: |z|={st_tr['mean_abs_raw']:8.2f} sat={st_tr['saturated_frac']:.2f}"
              f" | N={N_EVAL}: |z|={st_ev['mean_abs_raw']:8.2f} "
              f"sat={st_ev['saturated_frac']:.2f}")
        del probe
        clear()

    dead = [r for r in rows if r["auroc_ood"] <= 0.55]
    live = [r for r in rows if r["auroc_ood"] > 0.55]
    verdicts = {
        "dead_cells_saturated_at_eval": all(
            r["eval_len"]["saturated_frac"] > 0.95 for r in dead) if dead else None,
        "live_cells_unsaturated_at_eval": all(
            r["eval_len"]["saturated_frac"] < 0.95 for r in live) if live else None,
        "mean_sat_frac_dead": (sum(r["eval_len"]["saturated_frac"] for r in dead)
                               / len(dead)) if dead else None,
        "mean_sat_frac_live": (sum(r["eval_len"]["saturated_frac"] for r in live)
                               / len(live)) if live else None,
        "hypothesis": "AUROC 0.500 cells are constant-output because the pre-clamp logit "
                      "leaves [-c, c] for every evaluation sequence",
    }
    print("\nverdicts:", verdicts)
    save("exp2b_saturation.json", {"cells": rows, "verdicts": verdicts,
                                   "config": {"d_model": D_MODEL, "N_train": N_TRAIN,
                                              "N_eval": N_EVAL, "strength": STRENGTH,
                                              "quick": a.quick}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
