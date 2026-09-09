"""Experiment 2: hyperparameter and precision ablations.

The reviewer's objection is that c, H and the compute precision were asserted rather than
chosen. Two studies answer that:

  (A) A 2D grid over the bounding link c in {1, 2, 5, 10} and head count H in
      {1, 4, 8, 16, 32}. For each cell we record out-of-distribution AUROC (train at
      N=512, evaluate at N=16,384), convergence speed, and gradient-norm stability. The
      hypothesis worth testing is not "bigger H is better" but that H trades detection
      against the subgradient sparsity of Theorem 3.1: support is at most H per step, so
      small H should train badly at weak signal and large H should cost calibration
      through a larger upward bias in the sum of maxima.

  (B) STE clamp behaviour in fp16, bf16 and fp32 under deliberate saturation. The paper
      claims the plain clamp zeroes gradients and the STE does not. That was measured
      once, in one dtype. Here it is measured in all three, together with the fraction of
      parameters still receiving non-zero gradient, which is the quantity that actually
      decides whether a saturated probe can recover.

  python experiments/exp2_ablations.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimax_probe import ProbeConfig                                     # noqa: E402
from experiments.common import (auroc, dev, clear, make_probe, save,        # noqa: E402
                                scores_iter, synth_iter, train_probe,
                                epochs_to_converge, make_attack_direction)

D_MODEL = 2048
N_TRAIN = 512
N_EVAL = 16_384
STRENGTH = 0.15          # weak signal: the regime where H actually matters
K = 8


def grid(quick: bool) -> list[dict]:
    cs = [1.0, 2.0, 5.0, 10.0]
    Hs = [1, 4, 8, 16, 32]
    if quick:
        cs, Hs = [1.0, 10.0], [1, 8, 32]
    atk = make_attack_direction(D_MODEL)
    n_pc = 32 if quick else 96
    eps = 6 if quick else 12
    rows: list[dict] = []
    n_eval = 24 if quick else 40

    def eval_auroc(probe) -> float:
        # Regenerated per cell from fixed seeds: identical data everywhere, one
        # sequence resident at a time.
        p = scores_iter(probe, synth_iter(n_eval, N_EVAL, K, atk, True, STRENGTH, 998))
        q = scores_iter(probe, synth_iter(n_eval, N_EVAL, 0, atk, False, STRENGTH, 999))
        return auroc(p, q)

    for c in cs:
        for H in Hs:
            cfg = ProbeConfig(d_model=D_MODEL, hidden=512, n_heads=H,
                              chunk_size=4096, logit_clamp=c,
                              straight_through_clamp=True)
            probe, hist = train_probe("multimax", cfg, atk, N_train=N_TRAIN, k_train=K,
                                      strength=STRENGTH, n_per_class=n_pc, epochs=eps,
                                      seed=7, track=True)
            au = eval_auroc(probe)
            gns = [h["grad_norm"] for h in hist]
            losses = [h["loss"] for h in hist]
            rows.append({
                "c": c, "H": H, "auroc_ood": au,
                "final_loss": losses[-1],
                "epochs_to_converge": epochs_to_converge(hist),
                "grad_norm_mean": sum(gns) / len(gns),
                # coefficient of variation: scale-free instability measure, so it is
                # comparable across H (whose gradient magnitude scales with H).
                "grad_norm_cv": (torch.tensor(gns).std() / torch.tensor(gns).mean()).item(),
                "history": hist,
            })
            print(f"  c={c:<5} H={H:<3} AUROC={au:.3f}  loss={losses[-1]:.4f}  "
                  f"gn={rows[-1]['grad_norm_mean']:.2e}  cv={rows[-1]['grad_norm_cv']:.3f}")
            del probe
            clear()
    return rows


def precision_study() -> list[dict]:
    """Saturate the logit on purpose, then ask whether gradient survives, per dtype.

    The target is set to the WRONG label so the loss is large and the pressure is to move
    the logit back across the clamp boundary. With target == prediction the BCE gradient
    vanishes on its own and the test would measure loss saturation, not the clamp: that
    error was made once already and inverted the conclusion.
    """
    rows = []
    for dtype_name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16),
                           ("fp32", torch.float32)):
        for ste in (True, False):
            cfg = ProbeConfig(d_model=D_MODEL, hidden=512, n_heads=8, chunk_size=4096,
                              compute_dtype=dt, reduce_dtype=torch.float32,
                              logit_clamp=10.0, straight_through_clamp=ste)
            torch.manual_seed(3)
            probe = make_probe("multimax", cfg).to(dev()).train()
            for m in probe.modules():
                if isinstance(m, torch.nn.Linear):
                    m.to(dt)
            # Large-magnitude input drives the sum of maxima far past the clamp.
            x = (torch.randn(2048, D_MODEL, device=dev()) * 6.0)
            probe.zero_grad(set_to_none=True)
            z = probe.logits(x)
            raw = float(probe.raw_logit(x).detach())
            loss = F.binary_cross_entropy_with_logits(
                z.reshape(1), torch.zeros(1, device=dev()))
            loss.backward()

            grads = [p.grad for p in probe.parameters() if p.grad is not None]
            total = sum(g.numel() for g in grads)
            nonzero = sum(int((g != 0).sum()) for g in grads)
            gn = sum(float(g.norm()) for g in grads)
            finite = all(bool(torch.isfinite(g).all()) for g in grads)
            rows.append({
                "dtype": dtype_name, "ste": ste,
                "raw_logit": raw, "clamped_logit": float(z.detach()),
                "grad_norm": gn,
                "nonzero_grad_frac": nonzero / max(total, 1),
                "all_finite": finite,
            })
            print(f"  {dtype_name} ste={str(ste):5s} raw={raw:9.2f} "
                  f"gn={gn:.3e} nonzero={nonzero / max(total,1):.3f} finite={finite}")
            del probe, x
            clear()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    print("== (A) bounding link c x head count H ==")
    g = grid(a.quick)
    print("\n== (B) STE clamp across precisions, under saturation ==")
    p = precision_study()

    best = max(g, key=lambda r: r["auroc_ood"])
    ste_on = [r for r in p if r["ste"]]
    ste_off = [r for r in p if not r["ste"]]
    # Report marginals, not a single arbitrary cell. An earlier version quoted "the H=1
    # AUROC" by taking the first matching row, which silently meant "H=1 AT THE SMALLEST
    # c" and made a clamp effect look like a head-count effect.
    by_c = {c: [r["auroc_ood"] for r in g if r["c"] == c] for c in sorted({r["c"] for r in g})}
    by_H = {H: [r["auroc_ood"] for r in g if r["H"] == H] for H in sorted({r["H"] for r in g})}
    verdicts = {
        "best_cell": {"c": best["c"], "H": best["H"], "auroc_ood": best["auroc_ood"]},
        "auroc_max_by_c": {str(k): max(v) for k, v in by_c.items()},
        "auroc_max_by_H": {str(k): max(v) for k, v in by_H.items()},
        "auroc_mean_by_c": {str(k): sum(v) / len(v) for k, v in by_c.items()},
        "auroc_mean_by_H": {str(k): sum(v) / len(v) for k, v in by_H.items()},
        "chance_cells": [{"c": r["c"], "H": r["H"]} for r in g if r["auroc_ood"] <= 0.55],
        "ste_on_min_grad_norm": min(r["grad_norm"] for r in ste_on),
        "ste_off_max_grad_norm": max(r["grad_norm"] for r in ste_off),
        "ste_rescues_all_dtypes": all(r["grad_norm"] > 0 for r in ste_on),
        "plain_clamp_dead_all_dtypes": all(r["grad_norm"] == 0 for r in ste_off),
        "all_finite": all(r["all_finite"] for r in p),
    }
    print("\nverdicts:", verdicts)
    save("exp2_ablations.json", {"grid": g, "precision": p, "verdicts": verdicts,
                                 "config": {"d_model": D_MODEL, "N_train": N_TRAIN,
                                            "N_eval": N_EVAL, "strength": STRENGTH,
                                            "k": K, "quick": a.quick}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
