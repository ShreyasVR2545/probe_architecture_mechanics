"""Experiment 9: bootstrap confidence intervals and standard baselines.

Two gaps the paper had. Every real-residual number was a single point estimate on 24
held-out sequences per class, with no uncertainty attached, and the only comparators were
other aggregators. A reader could not tell whether a 0.04 AUROC gap was a result or noise,
nor whether any of this beats a linear probe.

This adds both.

  Metrics      AUROC and TPR at 1% FPR, each with a 95% CI from 1,000 bootstrap
               resamples of the held-out set, stratified by class so every resample keeps
               the original class balance. TPR@1%FPR is the operational number: a
               guardrail runs at a fixed, low false-positive budget, and AUROC averages
               over operating points nobody deploys at.

  Baselines    mean_logreg   logistic regression on the mean-pooled residual
               wlda          Fisher linear discriminant on the mean-pooled residual
               latentbiopsy  angular anomaly score: cosine deviation of the sequence
                             representation from a normative (benign) subspace fitted by
                             PCA on training benign activations
               multimax      hard-max aggregation, H=8
               topr          streaming top-r, r=8

A note on the angular baseline, because the choice matters and could be a strawman if
made carelessly. LatentBiopsy-style detectors score how far an activation sits from a
normative subspace. That score has to be formed over the sequence somehow, and the
natural, and the published, formulation compares the SEQUENCE representation to the
norm, which is a mean-like reduction and therefore dilutes with N. A max-over-tokens
angular variant would not dilute, but it would be our own aggregator wearing a different
score function, so it tests nothing. We implement the mean-form and say so, rather than
implement the max-form and claim a win.

  python experiments/exp9_bootstrap_baselines.py [--quick] [--n-bootstraps 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimax_probe import ProbeConfig, anneal_tau                          # noqa: E402
from experiments.common import auroc, clear, dev, make_probe, save          # noqa: E402
from experiments.exp1_real_residuals import CACHE as MISTRAL_CACHE, HARMFUL  # noqa: E402
from experiments.exp8_qwen_family import CACHE as QWEN_CACHE                # noqa: E402

LENGTHS = [512, 1024, 2048, 4096]
MODELS = {
    "mistral-7b": {"cache": MISTRAL_CACHE, "layers": (16, 24, 31), "d": 4096,
                   "depths": (0.50, 0.75, 1.00)},
    "qwen2.5-7b": {"cache": QWEN_CACHE, "layers": (13, 20, 27), "d": 3584,
                   "depths": (0.50, 0.75, 1.00)},
}
AGGREGATORS = ["mean_logreg", "wlda", "latentbiopsy", "multimax", "topr"]


# ======================================================================================
# metrics
# ======================================================================================
def tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, fpr: float = 0.01) -> float:
    """TPR at a threshold placed on the NEGATIVES at the given false-positive rate."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - fpr)
    return float((pos > thr).mean())


def auroc_np(pos: np.ndarray, neg: np.ndarray) -> float:
    return auroc(torch.from_numpy(pos).float(), torch.from_numpy(neg).float())


def bootstrap(pos: np.ndarray, neg: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    """Class-stratified bootstrap. Returns point estimates and 95% percentile CIs."""
    rng = np.random.default_rng(seed)
    au = [auroc_np(pos, neg)]
    tp = [tpr_at_fpr(pos, neg)]
    bau, btp = [], []
    for _ in range(n_boot):
        p = pos[rng.integers(0, pos.size, pos.size)]
        n = neg[rng.integers(0, neg.size, neg.size)]
        bau.append(auroc_np(p, n))
        btp.append(tpr_at_fpr(p, n))
    bau, btp = np.asarray(bau), np.asarray(btp)
    return {
        "auroc": au[0], "auroc_lo": float(np.percentile(bau, 2.5)),
        "auroc_hi": float(np.percentile(bau, 97.5)),
        "tpr_at_1fpr": tp[0], "tpr_lo": float(np.percentile(btp, 2.5)),
        "tpr_hi": float(np.percentile(btp, 97.5)),
        "n_boot": n_boot, "n_pos": int(pos.size), "n_neg": int(neg.size),
    }


def roc_curve(pos: np.ndarray, neg: np.ndarray, n_pts: int = 101):
    """FPR/TPR sampled on a fixed FPR grid so curves are averageable across resamples."""
    grid = np.linspace(0.0, 1.0, n_pts)
    if pos.size == 0 or neg.size == 0:
        return grid, np.full_like(grid, np.nan)
    thr = np.quantile(neg, 1.0 - grid)
    tpr = np.array([(pos > t).mean() for t in thr])
    return grid, tpr


# ======================================================================================
# data
# ======================================================================================
def load_split(cache: Path, N: int, layer: int, n_per_class: int):
    pos, neg = [], []
    for label, bucket in ((1, pos), (0, neg)):
        for i in range(n_per_class):
            f = cache / f"N{N}_y{label}_{i:03d}.pt"
            if f.exists():
                bucket.append((i, torch.load(f, map_location="cpu")[layer]))
    return pos, neg


def needle_split(items):
    """Needle-disjoint, matching the corrected exp1 protocol."""
    k = len(HARMFUL)
    return ([x for i, x in items if (i % k) < k // 2],
            [x for i, x in items if (i % k) >= k // 2])


# ======================================================================================
# scorers
# ======================================================================================
def _meanpool(seqs):
    return np.stack([t.float().mean(0).numpy() for t in seqs])


def score_mean_logreg(tr_p, tr_n, te_p, te_n):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X = np.concatenate([_meanpool(tr_p), _meanpool(tr_n)])
    y = np.concatenate([np.ones(len(tr_p)), np.zeros(len(tr_n))])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X, y)
    f = clf.decision_function
    return f(_meanpool(te_p)), f(_meanpool(te_n))


def score_wlda(tr_p, tr_n, te_p, te_n):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = np.concatenate([_meanpool(tr_p), _meanpool(tr_n)])
    y = np.concatenate([np.ones(len(tr_p)), np.zeros(len(tr_n))])
    # shrinkage is essential: d (3584-4096) far exceeds the sample count, so the
    # unregularised within-class scatter is singular and the solve is meaningless.
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(X, y)
    f = lda.decision_function
    return f(_meanpool(te_p)), f(_meanpool(te_n))


def score_latentbiopsy(tr_p, tr_n, te_p, te_n, k: int = 16):
    """Angular deviation of the sequence representation from a normative subspace.

    The subspace is fitted by PCA on BENIGN training activations only, which is what
    makes it an anomaly detector rather than a classifier: it never sees a positive.
    """
    from sklearn.decomposition import PCA
    Xn = _meanpool(tr_n)
    mu = Xn.mean(0, keepdims=True)
    pca = PCA(n_components=min(k, max(1, Xn.shape[0] - 1)))
    pca.fit(Xn - mu)
    B = pca.components_                                   # (k, d) orthonormal rows

    def dev(seqs):
        Z = _meanpool(seqs) - mu
        proj = Z @ B.T @ B                                # component inside the subspace
        num = np.linalg.norm(proj, axis=1)
        den = np.linalg.norm(Z, axis=1) + 1e-9
        cos = np.clip(num / den, 0.0, 1.0)                # cosine to the subspace
        return 1.0 - cos                                  # larger = more anomalous
    return dev(te_p), dev(te_n)


def score_probe(kind, tr_p, tr_n, te_p, te_n, d_model, epochs, r=8, seed=31):
    cfg = ProbeConfig(d_model=d_model, hidden=512, n_heads=8, chunk_size=4096)
    torch.manual_seed(seed)
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
            out.append(float(probe.logits(g).float().reshape(-1)[0]))
            del g
        return np.asarray(out)

    p, n = sc(te_p), sc(te_n)
    del probe
    clear()
    return p, n


SCORERS = {
    "mean_logreg": lambda a, b, c, d, dm, ep: score_mean_logreg(a, b, c, d),
    "wlda": lambda a, b, c, d, dm, ep: score_wlda(a, b, c, d),
    "latentbiopsy": lambda a, b, c, d, dm, ep: score_latentbiopsy(a, b, c, d),
    "multimax": lambda a, b, c, d, dm, ep: score_probe("multimax", a, b, c, d, dm, ep),
    "topr": lambda a, b, c, d, dm, ep: score_probe("topr", a, b, c, d, dm, ep),
}


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-bootstraps", type=int, default=1000)
    ap.add_argument("--models", default="mistral-7b,qwen2.5-7b")
    a = ap.parse_args()
    n_boot = 100 if a.quick else a.n_bootstraps
    epochs = 4 if a.quick else 10
    n_pc = 48
    lengths = [512, 4096] if a.quick else LENGTHS
    want = [m.strip() for m in a.models.split(",")]

    rows, curves, missing = [], [], []
    for mname in want:
        spec = MODELS[mname]
        for N in lengths:
            for li, layer in enumerate(spec["layers"]):
                pos, neg = load_split(spec["cache"], N, layer, n_pc)
                if len(pos) < 8 or len(neg) < 8:
                    missing.append({"model": mname, "N": N, "layer": layer})
                    continue
                tr_p, te_p = needle_split(pos)
                tr_n, te_n = needle_split(neg)
                for agg in AGGREGATORS:
                    sp, sn = SCORERS[agg](tr_p, tr_n, te_p, te_n, spec["d"], epochs)
                    st = bootstrap(np.asarray(sp), np.asarray(sn), n_boot, seed=17)
                    rows.append({"model": mname, "N": N, "layer": layer,
                                 "depth_ratio": spec["depths"][li],
                                 "aggregator": agg, **st})
                    fp, tp = roc_curve(np.asarray(sp), np.asarray(sn))
                    curves.append({"model": mname, "N": N, "layer": layer,
                                   "aggregator": agg, "fpr": fp.tolist(),
                                   "tpr": tp.tolist()})
                    print(f"  {mname:11s} N={N:<5} L{layer:<3} {agg:<13} "
                          f"AUROC={st['auroc']:.3f} "
                          f"[{st['auroc_lo']:.3f},{st['auroc_hi']:.3f}]  "
                          f"TPR@1%={st['tpr_at_1fpr']:.3f} "
                          f"[{st['tpr_lo']:.3f},{st['tpr_hi']:.3f}]")
                del pos, neg, tr_p, te_p, tr_n, te_n
                clear()

    def mean_over(model, agg, key="auroc"):
        v = [r[key] for r in rows if r["model"] == model and r["aggregator"] == agg]
        return float(np.mean(v)) if v else None

    def slope(model, agg, key="auroc"):
        """Change in the metric from the shortest to the longest length, averaged
        over layers. Negative means the score dilutes as the context grows."""
        lo = [r[key] for r in rows if r["model"] == model and r["aggregator"] == agg
              and r["N"] == min(lengths)]
        hi = [r[key] for r in rows if r["model"] == model and r["aggregator"] == agg
              and r["N"] == max(lengths)]
        if not lo or not hi:
            return None
        return float(np.mean(hi) - np.mean(lo))

    verdicts = {
        "n_bootstraps": n_boot, "lengths": lengths, "aggregators": AGGREGATORS,
        "split": "needle-disjoint",
        "mean_auroc": {m: {g: mean_over(m, g) for g in AGGREGATORS} for m in want},
        "mean_tpr_at_1fpr": {m: {g: mean_over(m, g, "tpr_at_1fpr")
                                 for g in AGGREGATORS} for m in want},
        "auroc_change_short_to_long": {m: {g: slope(m, g) for g in AGGREGATORS}
                                       for m in want},
        "missing_cells": missing,
    }
    print("\nverdicts:", json.dumps(verdicts["mean_auroc"], indent=1))
    print("dilution (AUROC change, shortest -> longest):",
          json.dumps(verdicts["auroc_change_short_to_long"], indent=1))
    save("exp9_bootstrap_baselines.json",
         {"rows": rows, "roc_curves": curves, "verdicts": verdicts,
          "config": {"n_per_class": n_pc, "epochs": epochs, "quick": a.quick,
                     "models": want}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
