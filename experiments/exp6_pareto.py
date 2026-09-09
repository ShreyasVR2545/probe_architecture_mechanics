"""Experiment 6: the accuracy-versus-memory Pareto frontier over (r, m) and depth.

Sweeps the two design parameters that actually trade detection against footprint, on REAL
Mistral-7B residual streams cached by exp1:

  Top-r        r in {1, 2, 4, 8, 16, 32}   carried state is Theta(r*H); r=1 is the plain
                                           hard maximum of the paper
  width m      m in {16, 32, 64, 128, 256} the token transform phi: R^d -> R^m, so m sets
                                           both the per-chunk activation and the parameter
                                           count
  depth        layer 16 (intermediate, 50% of 32) and layer 31 (near-final, 100%)

A note on `m`. The brief calls this the "sub-window size". In this paper `m` denotes the
output width of phi (Eq. 4), which is the quantity that appears in the memory law and in
the Theta(Nmd) FLOP count, so that is what is swept here. Chunk width, the other candidate
reading, is already swept in Suite D and is reported there as Theta(min(C,N)). If the
sub-window reading was intended, this sweep answers a different question than asked and
should be re-run; the interpretation is recorded in the artifact so the difference is not
silently buried.

Splits are needle-disjoint throughout, for the reason exp1 documents: an index split
shares needle sentences between train and test and inflates AUROC by about 0.1.

  python experiments/exp6_pareto.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimax_probe import ProbeConfig, anneal_tau                           # noqa: E402
from experiments.common import (ART, auroc, clear, dev, make_probe, peak_mib,  # noqa: E402
                                recall_at_fpr, save)
from experiments.exp1_real_residuals import CACHE, HARMFUL, LAYERS            # noqa: E402

N_ACT = 1024          # cached length used for the accuracy axis
N_MEM = 16_384        # length at which the memory axis is measured
D_MODEL = 4096
H = 8


def load_layer(layer: int, n_per_class: int):
    pos, neg = [], []
    for label, bucket in ((1, pos), (0, neg)):
        for i in range(n_per_class):
            f = CACHE / f"N{N_ACT}_y{label}_{i:03d}.pt"
            if f.exists():
                bucket.append((i, torch.load(f, map_location="cpu")[layer]))
    return pos, neg


def needle_split(items):
    """Hold out whole needle sentences: ids 0..5 train, 6..11 test."""
    k = len(HARMFUL)
    tr = [x for i, x in items if (i % k) < k // 2]
    te = [x for i, x in items if (i % k) >= k // 2]
    return tr, te


def train_eval(kind: str, r: int, m: int, tr_p, tr_n, te_p, te_n, epochs: int):
    cfg = ProbeConfig(d_model=D_MODEL, hidden=m, n_heads=H, chunk_size=4096)
    torch.manual_seed(19)
    probe = make_probe(kind, cfg, r=r).to(dev()).train()
    X = [t.to(dev(), torch.float32) for t in tr_p + tr_n]
    y = torch.tensor([1.0] * len(tr_p) + [0.0] * len(tr_n), device=dev())
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    can = hasattr(probe, "set_tau")
    for ep in range(epochs):
        if can:
            probe.set_tau(anneal_tau(ep, epochs))
        perm = torch.randperm(len(X))
        for s in range(0, len(X), 8):
            b = perm[s:s + 8]
            opt.zero_grad(set_to_none=True)
            z = torch.stack([probe.logits(X[int(j)]) for j in b])
            F.binary_cross_entropy_with_logits(z, y[b]).backward()
            opt.step()
    if can:
        probe.set_tau(0.0)
    probe.eval()
    del X
    clear()

    @torch.no_grad()
    def sc(items):
        out = []
        for t in items:
            g = t.to(dev(), torch.float32)
            out.append(probe.logits(g).float().reshape(-1)[0])
            del g
        return torch.stack(out)

    sp, sn = sc(te_p), sc(te_n)
    au, rc = auroc(sp, sn), recall_at_fpr(sp, sn, 0.01)
    n_par = probe.n_params()

    # memory axis: peak activation overhead above the input, at a long context
    probe_i = make_probe(kind, cfg, r=r).to(dev()).eval().to_compute_dtype()
    x = torch.randn(N_MEM, D_MODEL, device=dev(), dtype=torch.bfloat16)
    with torch.no_grad():
        _, pk = peak_mib(lambda: probe_i.logits(x))
    del x, probe_i, probe
    clear()
    return {"auroc": au, "recall_at_1pct_fpr": rc, "n_params": n_par,
            "overhead_mib": pk}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    rs = [1, 2, 4, 8, 16, 32] if not a.quick else [1, 8, 32]
    ms = [16, 32, 64, 128, 256] if not a.quick else [16, 128]
    layers = [16, 31]
    epochs = 4 if a.quick else 10
    n_pc = 48

    rows = []
    for layer in layers:
        pos, neg = load_layer(layer, n_pc)
        if len(pos) < 8:
            print(f"  layer {layer}: no cached activations at N={N_ACT}, skipping")
            continue
        tr_p, te_p = needle_split(pos)
        tr_n, te_n = needle_split(neg)
        print(f"  layer {layer}: train {len(tr_p)}+{len(tr_n)}, "
              f"test {len(te_p)}+{len(te_n)}")
        for m in ms:
            for r in rs:
                # r=1 IS the plain hard maximum, so route it through MultiMax rather than
                # Top-r with r=1: same function, but it exercises the audited module.
                kind = "multimax" if r == 1 else "topr"
                res = train_eval(kind, r, m, tr_p, tr_n, te_p, te_n, epochs)
                rows.append({"layer": layer, "r": r, "m": m, "probe": kind, **res})
                print(f"    L{layer} r={r:<3} m={m:<4} AUROC={res['auroc']:.3f} "
                      f"rec={res['recall_at_1pct_fpr']:.3f} "
                      f"mem={res['overhead_mib']:7.2f} MiB "
                      f"params={res['n_params']:>8}")
        del pos, neg, tr_p, te_p, tr_n, te_n
        clear()

    # Pareto front per layer: maximise AUROC, minimise memory
    def front(sub):
        pts = sorted(sub, key=lambda z: (z["overhead_mib"], -z["auroc"]))
        best, out = -1.0, []
        for p in pts:
            if p["auroc"] > best:
                out.append({"r": p["r"], "m": p["m"], "auroc": p["auroc"],
                            "overhead_mib": p["overhead_mib"]})
                best = p["auroc"]
        return out

    verdicts = {
        "m_interpretation": "phi output width (paper Eq. 4), not chunk/sub-window size",
        "pareto_front_by_layer": {str(L): front([r for r in rows if r["layer"] == L])
                                  for L in layers},
        "best_overall": max(rows, key=lambda r: r["auroc"]) if rows else None,
        "best_r1_baseline": max([r for r in rows if r["r"] == 1],
                                key=lambda r: r["auroc"], default=None),
    }
    if rows:
        for L in layers:
            sub = [r for r in rows if r["layer"] == L]
            if sub:
                verdicts[f"layer{L}_best_auroc"] = max(r["auroc"] for r in sub)
    print("\nverdicts:", verdicts.get("pareto_front_by_layer"))
    save("exp6_pareto.json", {"grid": rows, "verdicts": verdicts,
                              "config": {"N_acc": N_ACT, "N_mem": N_MEM, "H": H,
                                         "d_model": D_MODEL, "layers": layers,
                                         "r_values": rs, "m_values": ms,
                                         "split": "needle-disjoint",
                                         "quick": a.quick}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
