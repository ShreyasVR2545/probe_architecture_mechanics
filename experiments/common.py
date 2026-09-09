"""Shared harness for the reviewer-response experiments.

Everything the four experiment scripts need in common: device handling, the attack
geometry, a training loop that works for both the original probes and the extended ones,
AUROC, and artifact writing.

The attack construction is imported from benchmark_suite rather than reimplemented. That
matters: an earlier run used a unit-NORM attack direction, which spreads its magnitude
over d coordinates and is invisible against the background, and it produced a MultiMax
AUROC of 0.484 that looked like a scientific result and was an artifact of the generator.
Reusing the audited constructor keeps every new number on the same footing as the ones
already in the paper.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

from multimax_probe import ProbeConfig, build_probe, anneal_tau            # noqa: E402
from benchmark_suite import make_attack_direction, synth, dev, clear, sync, is_oom  # noqa: E402
from experiments.probes_ext import build_ext_probe                          # noqa: E402

BUILTIN = {"multimax", "softmax_attn", "self_attn", "mean_pool"}
EXTENDED = {"topr", "mean_max"}


def make_probe(kind: str, cfg: ProbeConfig, r: int = 8):
    if kind in BUILTIN:
        return build_probe(kind, cfg)
    return build_ext_probe(kind, cfg, r=r)


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Rank-based AUROC with tie correction (average ranks)."""
    pos = pos.float().flatten()
    neg = neg.float().flatten()
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    order = allv.argsort()
    ranks = torch.empty_like(allv)
    ranks[order] = torch.arange(1, allv.numel() + 1, dtype=allv.dtype,
                                device=allv.device)
    # average ranks within tie groups
    uniq, inv, cnt = allv.unique(return_inverse=True, return_counts=True)
    sums = torch.zeros_like(uniq).scatter_add_(0, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = pos.numel(), neg.numel()
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at_fpr(pos: torch.Tensor, neg: torch.Tensor, fpr: float = 0.01) -> float:
    """Recall at a threshold set on the NEGATIVES, which is how a guardrail is tuned."""
    pos, neg = pos.float().flatten(), neg.float().flatten()
    if neg.numel() == 0:
        return float("nan")
    thr = torch.quantile(neg, 1.0 - fpr)
    return float((pos > thr).float().mean())


@torch.no_grad()
def scores_of(probe, seqs: list[torch.Tensor]) -> torch.Tensor:
    """Score a list of sequences, staging CPU-resident ones to the GPU one at a time.

    Evaluation sets are large: 48 sequences of (16384, 2048) in fp32 is 6.4 GiB, and a
    positive and a negative set held simultaneously exceed this 8 GiB card. Holding them
    in host memory and moving one sequence at a time keeps peak VRAM at a single sequence
    while still letting the SAME eval set be reused across every cell of an ablation
    grid, which is what makes the cells comparable.
    """
    out = []
    for s in seqs:
        staged = s.to(dev()) if s.device.type != torch.device(dev()).type else s
        out.append(probe.logits(staged).float().reshape(-1)[0])
        if staged is not s:
            del staged
    return torch.stack(out)


def synth_iter(n: int, N: int, k: int, atk: torch.Tensor, positive: bool,
               strength: float, seed: int, contiguous: bool = True,
               split_energy: bool = False):
    """Generator form of benchmark_suite.synth: yields one sequence at a time.

    Mirrors synth's RNG consumption exactly, so for a given seed the i-th sequence here is
    the i-th sequence there. That equivalence is the point: an ablation grid needs every
    cell scored on identical data, but materialising the whole evaluation set costs
    48 x 128 MiB per class at N=16,384, which does not fit alongside a second class on an
    8 GiB card. Generating on demand keeps the data fixed and the peak at one sequence.
    """
    g = torch.Generator().manual_seed(seed)
    per_token = (strength / max(k, 1)) if split_energy else strength
    d = atk.numel()
    for _ in range(n):
        x = torch.randn(N, d, generator=g) * 0.5
        if positive and k > 0:
            if contiguous:
                j = int(torch.randint(0, max(N - k, 1), (1,), generator=g).item())
                idx = torch.arange(j, j + k)
            else:
                idx = torch.randperm(N, generator=g)[:k]
            x[idx] += per_token * atk.unsqueeze(0)
        yield x


@torch.no_grad()
def scores_iter(probe, it) -> torch.Tensor:
    """Score a generator of CPU sequences without ever holding two of them."""
    out = []
    for x in it:
        xg = x.to(dev())
        out.append(probe.logits(xg).float().reshape(-1)[0])
        del xg, x
    return torch.stack(out)


def train_probe(kind: str, cfg: ProbeConfig, atk: torch.Tensor, *,
                N_train: int = 512, k_train: int = 8, strength: float = 0.30,
                n_per_class: int = 96, epochs: int = 12, seed: int = 0,
                contiguous: bool = True, split_energy: bool = False,
                anneal: bool = True, r: int = 8, lr: float = 1e-3,
                track: bool = False):
    """Train on short sequences. Long context stays strictly out of distribution.

    With track=True the per-epoch loss and total gradient norm are returned alongside the
    probe, which is what the c x H and precision ablations report as "gradient stability"
    and "convergence speed".
    """
    torch.manual_seed(seed)
    probe = make_probe(kind, cfg, r=r).to(dev()).train()
    pos = synth(n_per_class, N_train, k_train, atk, True, strength, seed, dev(),
                contiguous, split_energy)
    neg = synth(n_per_class, N_train, 0, atk, False, strength, seed + 1, dev(),
                contiguous, split_energy)
    X, y = pos + neg, torch.tensor([1.0] * len(pos) + [0.0] * len(neg), device=dev())
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)
    can_anneal = anneal and hasattr(probe, "set_tau")
    hist: list[dict] = []

    for ep in range(epochs):
        if can_anneal:
            probe.set_tau(anneal_tau(ep, epochs))
        perm = torch.randperm(len(X))
        ep_loss, ep_gn, nb = 0.0, 0.0, 0
        for s in range(0, len(X), 32):
            b = perm[s:s + 32]
            opt.zero_grad(set_to_none=True)
            z = torch.stack([probe.logits(X[j]) for j in b])
            loss = F.binary_cross_entropy_with_logits(z, y[b])
            loss.backward()
            if track:
                gn = sum(float(p.grad.norm()) for p in probe.parameters()
                         if p.grad is not None)
                ep_gn += gn
            ep_loss += float(loss.detach())
            nb += 1
            opt.step()
        if track:
            hist.append({"epoch": ep, "loss": ep_loss / max(nb, 1),
                         "grad_norm": ep_gn / max(nb, 1)})
    if can_anneal:
        probe.set_tau(0.0)          # deploy the exact hard max
    probe.eval()
    del X, pos, neg
    clear()
    return (probe, hist) if track else probe


def epochs_to_converge(hist: list[dict], frac: float = 0.10) -> int | None:
    """First epoch whose mean loss is within `frac` of the best achieved. None if never."""
    if not hist:
        return None
    best = min(h["loss"] for h in hist)
    for h in hist:
        if h["loss"] <= best * (1.0 + frac):
            return int(h["epoch"])
    return None


def save(name: str, payload: dict) -> Path:
    """Write one experiment artifact, with provenance, into artifacts/."""
    ART.mkdir(exist_ok=True)
    payload = dict(payload)
    payload["_meta"] = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_total_gib": (torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                           if torch.cuda.is_available() else None),
    }
    p = ART / name
    p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT)}")
    return p


def peak_mib(fn, *a, **kw):
    """Run fn and return (result, peak allocated MiB above the pre-call baseline)."""
    if not torch.cuda.is_available():
        return fn(*a, **kw), float("nan")
    clear()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn(*a, **kw)
    sync()
    peak = torch.cuda.max_memory_allocated()
    return out, (peak - base) / 2 ** 20


def timed(fn, *a, n_warmup: int = 2, n_iter: int = 5, **kw) -> float:
    """Median wall time in ms over n_iter runs after n_warmup warmups."""
    for _ in range(n_warmup):
        fn(*a, **kw)
    sync()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(*a, **kw)
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]
