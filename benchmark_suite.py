r"""
benchmark_suite.py — latency, memory, and localized-attack benchmarks.

Two suites, both run on synthetic hidden states so that sequence length is a free
parameter rather than being capped by what the host LLM can hold on an 8 GiB card.

SUITE A — context-length latency & memory
    N in {128, 1024, 8192, 32768, 131072}, d = 2048.
    Probes: multimax (chunked hard-max), softmax_attn (single-query pooling),
            mean_pool, self_attn (full N x N attention).
    Records wall-clock forward latency and peak CUDA memory, and marks OOM rather
    than aborting.

    On the O(N^2) question: single-query attention pooling is O(N), not O(N^2) --
    its score tensor is (B, N). Only self_attn is quadratic. The suite measures all
    four so the scaling claim is read off data instead of asserted.

SUITE B — localized attack insertion (context dilution)
    Baseline benign sequences of length N = 16384. A k-token attack signal,
    k in {4, 8, 16}, is spliced in at a position drawn uniformly from [0, N-k].
    Probes are trained on SHORT clean sequences and never see a long one during
    training, which is exactly the production failure mode: train short, deploy long.

    FNR is measured at a threshold calibrated to 1% FPR on held-out benign sequences
    of the *same length*, so the comparison is not contaminated by a length-dependent
    score shift.

Run:  python benchmark_suite.py            (full)
      python benchmark_suite.py --quick    (smaller N ladder, fewer trials)
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from multimax_probe import ProbeConfig, build_probe

ROOT = Path(__file__).resolve().parent
GIB = 1024 ** 3
D_MODEL = 2048
LADDER_FULL = (128, 1024, 8192, 32768, 131072)
LADDER_QUICK = (128, 1024, 8192, 32768)
PROBE_KINDS = ("multimax", "softmax_attn", "mean_pool", "self_attn")
SELF_ATTN_MAX_N = 8192          # beyond this the N x N score matrix cannot fit; still attempted


def dev() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError)


# ======================================================================================
# Suite A — latency & memory
# ======================================================================================
def suite_a(ladder: tuple[int, ...], n_warmup: int = 2, n_iter: int = 5) -> list[dict]:
    print("=" * 86)
    print("SUITE A — context-length latency & peak memory  (batch=1, d=%d)" % D_MODEL)
    print("=" * 86)
    print(f"  {'probe':<14s}{'N':>9s}{'latency ms':>13s}{'ms/1k tok':>12s}"
          f"{'peak MiB':>11s}{'status':>12s}")
    rows: list[dict] = []
    for kind in PROBE_KINDS:
        cfg = ProbeConfig(d_model=D_MODEL, hidden=512, n_heads=8, chunk_size=4096)
        probe = build_probe(kind, cfg).to(dev()).to_compute_dtype().eval()
        for N in ladder:
            clear()
            rec = {"probe": kind, "N": N, "d_model": D_MODEL}
            try:
                x = torch.randn(1, N, D_MODEL, device=dev(), dtype=torch.bfloat16)
                with torch.no_grad():
                    for _ in range(n_warmup):
                        probe.logits(x)
                    sync()
                    ts = []
                    for _ in range(n_iter):
                        t0 = time.perf_counter()
                        probe.logits(x)
                        sync()
                        ts.append((time.perf_counter() - t0) * 1000.0)
                ts.sort()
                ms = ts[len(ts) // 2]                       # median
                peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
                rec |= {"latency_ms": round(ms, 4), "ms_per_1k_tokens": round(ms / (N / 1000), 5),
                        "peak_mib": round(peak, 1), "status": "ok"}
                print(f"  {kind:<14s}{N:>9d}{ms:>13.3f}{ms/(N/1000):>12.4f}"
                      f"{peak:>11.1f}{'ok':>12s}")
                del x
            except Exception as e:
                if not is_oom(e):
                    raise
                rec |= {"latency_ms": None, "ms_per_1k_tokens": None, "peak_mib": None,
                        "status": "OOM"}
                print(f"  {kind:<14s}{N:>9d}{'-':>13s}{'-':>12s}{'-':>11s}{'OOM':>12s}")
                clear()
            rows.append(rec)
        del probe
        clear()

    # scaling exponent: fit log(peak_mem) ~ alpha * log(N)
    print(f"\n  {'probe':<14s}{'mem scaling alpha':>20s}{'interpretation':>28s}")
    for kind in PROBE_KINDS:
        pts = [(r["N"], r["peak_mib"]) for r in rows
               if r["probe"] == kind and r.get("peak_mib")]
        if len(pts) >= 2:
            import math
            xs = [math.log(n) for n, _ in pts]
            ys = [math.log(m) for _, m in pts]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = sum((a - mx) ** 2 for a in xs) or 1e-9
            alpha = num / den
            interp = ("~O(1) in N" if alpha < 0.35 else
                      "~O(N)" if alpha < 1.4 else "~O(N^2)")
            print(f"  {kind:<14s}{alpha:>20.3f}{interp:>28s}")
            for r in rows:
                if r["probe"] == kind:
                    r["mem_scaling_alpha"] = round(alpha, 4)
    return rows


# ======================================================================================
# Suite B — localized attack insertion
# ======================================================================================
def make_attack_direction(d: int, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return v / v.norm()


def synth(n: int, N: int, k: int, atk: torch.Tensor, positive: bool,
          strength: float, seed: int, device: str) -> list[torch.Tensor]:
    """Benign N(0, 0.5) background; positives get k consecutive attack tokens."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(N, atk.numel(), generator=g) * 0.5
        if positive and k > 0:
            j = int(torch.randint(0, max(N - k, 1), (1,), generator=g).item())
            x[j:j + k] += strength * atk.unsqueeze(0)
        out.append(x.to(device))
    return out


def train_probe_short(kind: str, cfg: ProbeConfig, atk: torch.Tensor,
                      N_train: int, k_train: int, strength: float,
                      n_per_class: int, epochs: int, seed: int) -> object:
    """Train on SHORT sequences only. Long context is strictly out-of-distribution."""
    torch.manual_seed(seed)
    probe = build_probe(kind, cfg).to(dev()).train()
    pos = synth(n_per_class, N_train, k_train, atk, True, strength, seed, dev())
    neg = synth(n_per_class, N_train, 0, atk, False, strength, seed + 1, dev())
    X = pos + neg
    y = torch.tensor([1.0] * len(pos) + [0.0] * len(neg), device=dev())
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, 32):
            b = perm[s:s + 32]
            opt.zero_grad(set_to_none=True)
            z = torch.stack([probe.logits(X[j]) for j in b])
            F.binary_cross_entropy_with_logits(z, y[b]).backward()
            opt.step()
    probe.eval()
    del X, pos, neg
    clear()
    return probe


@torch.no_grad()
def scores_of(probe, seqs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([probe.logits(s).reshape(-1)[0].float().cpu() for s in seqs])


def suite_b(quick: bool) -> list[dict]:
    print("\n" + "=" * 86)
    print("SUITE B — localized attack insertion (context dilution)")
    print("=" * 86)
    N_train, k_train, strength = 512, 8, 3.0
    N_test = 16384 if not quick else 8192
    ks = (4, 8, 16)
    n_eval = 60 if not quick else 30
    atk = make_attack_direction(D_MODEL)

    kinds = ("multimax", "softmax_attn", "mean_pool")
    print(f"  train: N={N_train}, k={k_train}, {2*(120 if not quick else 60)} seqs "
          f"| test: N={N_test}, {n_eval} pos + {n_eval} neg per k")
    print(f"  FNR measured at a threshold giving 1% FPR on benign of the SAME length\n")

    rows: list[dict] = []
    probes = {}
    for kind in kinds:
        cfg = ProbeConfig(d_model=D_MODEL, hidden=512, n_heads=8, chunk_size=4096)
        probes[kind] = train_probe_short(kind, cfg, atk, N_train, k_train, strength,
                                         120 if not quick else 60, 25, seed=0)
        print(f"  trained {kind}")

    print(f"\n  {'probe':<14s}{'k':>4s}{'N':>8s}{'recall@1%FPR':>15s}{'FNR':>9s}"
          f"{'AUROC':>9s}")
    for kind in kinds:
        probe = probes[kind]
        # in-distribution reference at the training length
        neg_s_short = scores_of(probe, synth(n_eval, N_train, 0, atk, False, strength, 91, dev()))
        pos_s_short = scores_of(probe, synth(n_eval, N_train, k_train, atk, True, strength, 92, dev()))
        thr_short = torch.quantile(neg_s_short, 0.99)
        rec_short = (pos_s_short > thr_short).float().mean().item()
        rows.append({"probe": kind, "k": k_train, "N": N_train, "recall": rec_short,
                     "fnr": 1 - rec_short, "auroc": _auroc(pos_s_short, neg_s_short),
                     "regime": "in-distribution"})
        print(f"  {kind:<14s}{k_train:>4d}{N_train:>8d}{rec_short:>15.4f}"
              f"{1-rec_short:>9.4f}{rows[-1]['auroc']:>9.4f}   <- train length")

        neg_s = scores_of(probe, synth(n_eval, N_test, 0, atk, False, strength, 7, dev()))
        thr = torch.quantile(neg_s, 0.99)
        for k in ks:
            pos_s = scores_of(probe, synth(n_eval, N_test, k, atk, True, strength, 8 + k, dev()))
            rec = (pos_s > thr).float().mean().item()
            au = _auroc(pos_s, neg_s)
            rows.append({"probe": kind, "k": k, "N": N_test, "recall": rec,
                         "fnr": 1 - rec, "auroc": au, "regime": "long-context OOD",
                         "dilution_ratio": k / N_test})
            print(f"  {kind:<14s}{k:>4d}{N_test:>8d}{rec:>15.4f}{1-rec:>9.4f}{au:>9.4f}")
        clear()
    return rows


def _auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    s = torch.cat([pos, neg])
    y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    ladder = LADDER_QUICK if args.quick else LADDER_FULL
    env = {
        "python": platform.python_version(), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_total_gib": round(torch.cuda.mem_get_info()[1] / GIB, 2)
        if torch.cuda.is_available() else 0.0,
        "d_model": D_MODEL, "quick": args.quick,
    }
    print(f"env: {env}\n")

    a = suite_a(ladder)
    b = suite_b(args.quick)

    # ---- verdicts -------------------------------------------------------------------
    print("\n" + "=" * 86)
    print("VERDICTS")
    print("=" * 86)
    mm_long = [r for r in b if r["probe"] == "multimax" and r["regime"] == "long-context OOD"]
    mm_min_recall = min(r["recall"] for r in mm_long) if mm_long else 0.0
    claim = mm_min_recall > 0.99
    print(f"  MultiMax recall > 99% at every k under long-context OOD : "
          f"{'HOLDS' if claim else 'DOES NOT HOLD'}  (min recall {mm_min_recall:.4f})")
    for kind in ("softmax_attn", "mean_pool"):
        rs = [r["recall"] for r in b if r["probe"] == kind and r["regime"] == "long-context OOD"]
        if rs:
            print(f"  {kind:<14s} min recall under the same shift          : {min(rs):.4f}")
    alphas = {r["probe"]: r.get("mem_scaling_alpha") for r in a if r.get("mem_scaling_alpha")}
    print(f"  memory scaling exponents (log-log fit)                  : "
          + ", ".join(f"{k}={v:.2f}" for k, v in alphas.items()))

    out = {"env": env, "suite_a_latency_memory": a, "suite_b_attack_insertion": b,
           "verdicts": {"multimax_recall_above_99pct": bool(claim),
                        "multimax_min_recall_long_ctx": mm_min_recall,
                        "memory_scaling_alpha": alphas}}
    p = ROOT / "benchmark_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  -> wrote {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
