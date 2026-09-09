"""Experiment 4: unknown-m fragmentation, aggregator ensembles, and batched systems.

Two gaps the reviewer identified, both real:

  (A) Benchmark C in the paper trains on the SAME fragmentation width m that it
      evaluates, which is generous to the defender in exactly the way that matters. An
      adversary picks m; the defender does not know it. Here the probe is trained on a
      MIXTURE of widths and evaluated on held-out widths, including a mixed-structure
      adversary that combines a concentrated needle with a diffuse background smear. The
      candidate set is widened to include the two countermeasures section 7.2 proposed but
      never tested: streaming Top-r, and a Mean-Max ensemble.

      The prediction worth falsifying is that Mean-Max dominates. The max and the mean
      fail on disjoint geometries (fragmented budget vs concentrated needle), so their sum
      should be at least as good as either everywhere. If it is not, the ensembling
      recommendation in section 7.2 is wrong and should be withdrawn.

  (B) Every systems number in the paper is at batch size 1. Deployments batch. Peak memory
      is reported both including and excluding the input tensor, because the input is the
      part the serving stack already pays for: counting it makes every aggregator look
      similar and hides the actual difference between them.

  python experiments/exp4_fragmentation.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimax_probe import ProbeConfig                                       # noqa: E402
from experiments.common import (auroc, clear, dev, make_attack_direction,     # noqa: E402
                                make_probe, peak_mib, recall_at_fpr, save,
                                scores_of, timed, train_probe, synth)

D_MODEL = 2048
BUDGET = 2.0
N_EVAL = 16_384
N_TRAIN = 512
CANDIDATES = ("multimax", "softmax_attn", "mean_pool", "topr", "mean_max")


def frag_seqs(n: int, N: int, m: int, atk, positive: bool, seed: int,
              mixed: bool = False):
    """Fixed budget spread over m tokens. mixed=True splits it between two geometries.

    The per-token strength is BUDGET/m, so the TOTAL injected energy is constant in m.
    That is what makes this an attack on the maximum specifically rather than simply a
    weaker attack, and it matches the Benchmark C construction.

    The mixed adversary is built in one pass over a single background rather than by
    adding two generated sequences together. Summing two synth() outputs would also sum
    their independent backgrounds, raising the noise floor by sqrt(2) for positives only
    and handing the probe a trivial variance cue that has nothing to do with the attack.
    """
    g = torch.Generator().manual_seed(seed)
    d = atk.numel()
    out = []
    for _ in range(n):
        x = torch.randn(N, d, generator=g) * 0.5
        if positive:
            if not mixed:
                idx = torch.randperm(N, generator=g)[:m]
                x[idx] += (BUDGET / m) * atk.unsqueeze(0)
            else:
                # half the budget in a tight 4-token core, half smeared over 4m tokens
                k_core = 4
                j = int(torch.randint(0, max(N - k_core, 1), (1,), generator=g).item())
                x[torch.arange(j, j + k_core)] += ((BUDGET / 2) / k_core) * atk.unsqueeze(0)
                k_sm = min(4 * m, N)
                idx = torch.randperm(N, generator=g)[:k_sm]
                x[idx] += ((BUDGET / 2) / k_sm) * atk.unsqueeze(0)
        out.append(x.to(dev()))
    return out


def unknown_m(quick: bool) -> list[dict]:
    """Train on a mixture of widths, evaluate on each width and on mixed structure."""
    atk = make_attack_direction(D_MODEL)
    train_ms = [1, 8, 64]
    eval_ms = [1, 4, 16, 64, 256] if not quick else [1, 16, 256]
    n_pc = 32 if quick else 64
    epochs = 6 if quick else 12
    rows = []

    for kind in (CANDIDATES if not quick else ("multimax", "mean_pool", "mean_max")):
        cfg = ProbeConfig(d_model=D_MODEL, hidden=512, n_heads=8, chunk_size=4096)
        # Train set mixes widths so the probe cannot specialise to one m.
        torch.manual_seed(3)
        probe = make_probe(kind, cfg, r=8).to(dev()).train()
        import torch.nn.functional as F
        from multimax_probe import anneal_tau
        pos, neg = [], []
        for j, m in enumerate(train_ms):
            pos += frag_seqs(n_pc // len(train_ms), N_TRAIN, m, atk, True, 40 + j)
            neg += frag_seqs(n_pc // len(train_ms), N_TRAIN, m, atk, False, 80 + j)
        X = pos + neg
        y = torch.tensor([1.0] * len(pos) + [0.0] * len(neg), device=dev())
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
        can = hasattr(probe, "set_tau")
        for ep in range(epochs):
            if can:
                probe.set_tau(anneal_tau(ep, epochs))
            perm = torch.randperm(len(X))
            for s in range(0, len(X), 16):
                b = perm[s:s + 16]
                opt.zero_grad(set_to_none=True)
                z = torch.stack([probe.logits(X[j]) for j in b])
                F.binary_cross_entropy_with_logits(z, y[b]).backward()
                opt.step()
        if can:
            probe.set_tau(0.0)
        probe.eval()
        del X, pos, neg
        clear()

        for m in eval_ms:
            for mixed in (False, True):
                p = frag_seqs(n_pc // 2, N_EVAL, m, atk, True, 500 + m, mixed)
                q = frag_seqs(n_pc // 2, N_EVAL, m, atk, False, 900 + m, mixed)
                sp, sq = scores_of(probe, p), scores_of(probe, q)
                rows.append({"probe": kind, "m": m, "mixed_structure": mixed,
                             "auroc": auroc(sp, sq),
                             "recall_at_1pct_fpr": recall_at_fpr(sp, sq, 0.01),
                             "trained_on_m": train_ms, "held_out": m not in train_ms})
                print(f"  {kind:<13} m={m:<4} mixed={str(mixed):5s} "
                      f"AUROC={rows[-1]['auroc']:.3f} "
                      f"rec={rows[-1]['recall_at_1pct_fpr']:.3f}"
                      f"{'  [held out]' if m not in train_ms else ''}")
                del p, q
                clear()
        del probe
        clear()
    return rows


def batched_systems(quick: bool) -> list[dict]:
    Bs = [1, 2, 4, 8, 16] if not quick else [1, 4, 16]
    N = 65_536 if not quick else 16_384
    rows = []
    for B in Bs:
        for kind in ("multimax", "softmax_attn", "mean_pool"):
            cfg = ProbeConfig(d_model=D_MODEL, hidden=512, n_heads=8, chunk_size=4096)
            probe = make_probe(kind, cfg).to(dev()).eval().to_compute_dtype()
            try:
                x = torch.randn(B, N, D_MODEL, device=dev(), dtype=torch.bfloat16)
                inp = x.numel() * x.element_size() / 2 ** 20
                with torch.no_grad():
                    _, pk = peak_mib(lambda: probe.logits(x))
                    ms = timed(lambda: probe.logits(x), n_warmup=1, n_iter=3)
                rows.append({"B": B, "N": N, "probe": kind,
                             "input_mib": inp, "overhead_above_input_mib": pk,
                             "total_mib": inp + pk, "latency_ms": ms,
                             "overhead_per_seq_mib": pk / B, "oom": False})
                print(f"  B={B:<3} {kind:<13} input={inp:8.1f} "
                      f"overhead={pk:8.1f} ({pk / B:6.1f}/seq) {ms:8.2f} ms")
                del x
            except torch.cuda.OutOfMemoryError:
                rows.append({"B": B, "N": N, "probe": kind, "oom": True})
                print(f"  B={B:<3} {kind:<13} OOM")
            del probe
            clear()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    print("== (A) unknown-m and mixed-structure fragmentation ==")
    frag = unknown_m(a.quick)
    print("\n== (B) batched systems scaling ==")
    batch = batched_systems(a.quick)

    def best_at(m, mixed):
        c = [r for r in frag if r["m"] == m and r["mixed_structure"] is mixed]
        return max(c, key=lambda r: r["auroc"])["probe"] if c else None

    ms = sorted({r["m"] for r in frag})
    mm_dom = all(
        max((r["auroc"] for r in frag
             if r["probe"] == "mean_max" and r["m"] == m and r["mixed_structure"] is mx),
            default=-1)
        >= max((r["auroc"] for r in frag
                if r["probe"] in ("multimax", "mean_pool") and r["m"] == m
                and r["mixed_structure"] is mx), default=-1) - 1e-9
        for m in ms for mx in (False, True))
    verdicts = {
        "best_probe_per_m_plain": {m: best_at(m, False) for m in ms},
        "best_probe_per_m_mixed": {m: best_at(m, True) for m in ms},
        "mean_max_dominates_both_members": bool(mm_dom),
        "batched_multimax_overhead_per_seq_flat": None,
    }
    mmb = [r for r in batch if r["probe"] == "multimax" and not r.get("oom")]
    if len(mmb) > 1:
        per = [r["overhead_per_seq_mib"] for r in mmb]
        verdicts["batched_multimax_overhead_per_seq_flat"] = (max(per) - min(per)) < 1.0
        verdicts["batched_multimax_per_seq_range_mib"] = [min(per), max(per)]
    print("\nverdicts:", verdicts)
    save("exp4_fragmentation.json",
         {"fragmentation": frag, "batched": batch, "verdicts": verdicts,
          "config": {"d_model": D_MODEL, "budget": BUDGET, "N_eval": N_EVAL,
                     "N_train": N_TRAIN, "quick": a.quick}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
