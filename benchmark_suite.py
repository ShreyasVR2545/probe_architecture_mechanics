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

from multimax_probe import ProbeConfig, anneal_tau, build_probe

NL = chr(10)
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
          f"{'peak MiB':>11s}{'overhead':>12s}{'status':>10s}")
    print(f"  {'':<14s}{'':>9s}{'':>13s}{'':>12s}{'':>11s}"
          f"{'(above input)':>12s}")
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
                input_mib = N * D_MODEL * 2 / 1024 ** 2      # bf16 input, unavoidable O(N)
                overhead = max(peak - input_mib, 0.0)
                rec |= {"latency_ms": round(ms, 4), "ms_per_1k_tokens": round(ms / (N / 1000), 5),
                        "peak_mib": round(peak, 1), "input_mib": round(input_mib, 1),
                        "overhead_mib": round(overhead, 1), "status": "ok"}
                print(f"  {kind:<14s}{N:>9d}{ms:>13.3f}{ms/(N/1000):>12.4f}"
                      f"{peak:>11.1f}{overhead:>12.1f}{'ok':>10s}")
                del x
            except Exception as e:
                if not is_oom(e):
                    raise
                rec |= {"latency_ms": None, "ms_per_1k_tokens": None, "peak_mib": None,
                        "status": "OOM"}
                print(f"  {kind:<14s}{N:>9d}{'-':>13s}{'-':>12s}{'-':>11s}{'-':>12s}{'OOM':>10s}")
                clear()
            rows.append(rec)
        del probe
        clear()

    # scaling exponent: fit log(peak_mem) ~ alpha * log(N)
    print(f"\n  {'probe':<14s}{'mem scaling alpha':>20s}{'interpretation':>28s}")
    for kind in PROBE_KINDS:
        pts = [(r["N"], r["overhead_mib"]) for r in rows
               if r["probe"] == kind and r.get("overhead_mib") and r["N"] >= 8192]
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
            print(f"  {kind:<14s}{alpha:>10.3f}{interp:>28s}")
            for r in rows:
                if r["probe"] == kind:
                    r["mem_scaling_alpha"] = round(alpha, 4)
    return rows


# ======================================================================================
# Suite B — localized attack insertion
# ======================================================================================
def make_attack_direction(d: int, seed: int = 1234) -> torch.Tensor:
    """Unit direction scaled to sqrt(d), i.e. unit magnitude PER COORDINATE.

    A unit-NORM direction spreads its mass over d=2048 dimensions (~0.022 per
    coordinate) and is invisible against background sigma=0.5. Measured in a first run:
    MultiMax AUROC 0.484 at its own training length -- the max reduction was tracking
    benign extremes, exactly the failure mode of `rem:fpr` (Remark 2.8) in math_formulation.tex.
    Scaling by sqrt(d) puts the attack on the same per-coordinate scale as the noise.
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return (v / v.norm()) * (d ** 0.5)


def synth(n: int, N: int, k: int, atk: torch.Tensor, positive: bool,
          strength: float, seed: int, device: str,
          contiguous: bool = True, split_energy: bool = False) -> list[torch.Tensor]:
    """Benign N(0, 0.5) background; positives carry a k-token attack.

    contiguous=True   k consecutive tokens at a random offset (Benchmark B).
    contiguous=False  k tokens scattered uniformly at random positions (Benchmark C) --
                      the adversary who refuses to put the signal in one place.
    split_energy=True the TOTAL signal budget is held fixed and divided across the k
                      tokens, so each individual token carries strength/k. This is the
                      attack aimed squarely at a max reduction: the sum is unchanged, so
                      mean pooling sees exactly what it saw before, while the per-token
                      peak that the hard max depends on falls as 1/k.
    """
    g = torch.Generator().manual_seed(seed)
    per_token = (strength / max(k, 1)) if split_energy else strength
    out = []
    for _ in range(n):
        x = torch.randn(N, atk.numel(), generator=g) * 0.5
        if positive and k > 0:
            if contiguous:
                j = int(torch.randint(0, max(N - k, 1), (1,), generator=g).item())
                idx = torch.arange(j, j + k)
            else:
                idx = torch.randperm(N, generator=g)[:k]
            x[idx] += per_token * atk.unsqueeze(0)
        out.append(x.to(device))
    return out


def train_probe_short(kind: str, cfg: ProbeConfig, atk: torch.Tensor,
                      N_train: int, k_train: int, strength: float,
                      n_per_class: int, epochs: int, seed: int,
                      contiguous: bool = True, split_energy: bool = False,
                      anneal: bool = True) -> object:
    """Train on SHORT sequences only. Long context is strictly out-of-distribution.

    For MultiMax the smooth-max temperature is annealed from LogSumExp down to the hard
    max (see multimax_probe.anneal_tau). Without it the hard-max subgradient reaches only
    H tokens per step and the probe failed to learn at all at strength 0.10 -- an
    optimisation failure that a first run mistook for a dilution result.
    """
    torch.manual_seed(seed)
    probe = build_probe(kind, cfg).to(dev()).train()
    pos = synth(n_per_class, N_train, k_train, atk, True, strength, seed, dev(),
                contiguous, split_energy)
    neg = synth(n_per_class, N_train, 0, atk, False, strength, seed + 1, dev(),
                contiguous, split_energy)
    X = pos + neg
    y = torch.tensor([1.0] * len(pos) + [0.0] * len(neg), device=dev())
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(X)
    can_anneal = anneal and hasattr(probe, "set_tau")
    for ep in range(epochs):
        if can_anneal:
            probe.set_tau(anneal_tau(ep, epochs))
        perm = torch.randperm(n)
        for s in range(0, n, 32):
            b = perm[s:s + 32]
            opt.zero_grad(set_to_none=True)
            z = torch.stack([probe.logits(X[j]) for j in b])
            F.binary_cross_entropy_with_logits(z, y[b]).backward()
            opt.step()
    if can_anneal:
        probe.set_tau(0.0)          # deploy the exact hard max
    probe.eval()
    del X, pos, neg
    clear()
    return probe


@torch.no_grad()
def scores_of(probe, seqs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([probe.logits(s).reshape(-1)[0].float().cpu() for s in seqs])


def suite_b(quick: bool) -> list[dict]:
    """Contiguous needle, swept signal strength, dual recall (long / train length)."""
    print(NL + "=" * 86)
    print("SUITE B - contiguous needle, signal-strength sweep")
    print("=" * 86)
    N_train, k_train = 512, 8
    N_test = 16384 if not quick else 8192
    K = 4
    strengths = (0.10, 0.15, 0.50)
    n_eval = 50 if not quick else 25
    n_train = 100 if not quick else 60
    atk = make_attack_direction(D_MODEL)
    kinds = ("multimax", "softmax_attn", "mean_pool")

    print(f"  k={K} contiguous tokens in N={N_test} (dilution {K/N_test:.2e}), "
          f"trained at N={N_train}")
    print(f"  MultiMax trained WITH smooth-max annealing; recall @ 1% FPR, "
          f"{n_eval} pos + {n_eval} neg" + NL)
    print(f"  {'strength':>9s}" + "".join(f"{k:>18s}" for k in kinds))
    print(f"  {'':>9s}" + "".join(f"{'R@16384 / R@512':>18s}" for _ in kinds))

    rows: list[dict] = []
    for st in strengths:
        cells = []
        for kind in kinds:
            cfg = ProbeConfig(d_model=D_MODEL, hidden=512, n_heads=8, chunk_size=4096)
            probe = train_probe_short(kind, cfg, atk, N_train, k_train, st,
                                      n_per_class=n_train, epochs=20, seed=0)
            negL = scores_of(probe, synth(n_eval, N_test, 0, atk, False, st, 7, dev()))
            posL = scores_of(probe, synth(n_eval, N_test, K, atk, True, st, 8, dev()))
            recL = (posL > torch.quantile(negL, 0.99)).float().mean().item()
            negS = scores_of(probe, synth(n_eval, N_train, 0, atk, False, st, 11, dev()))
            posS = scores_of(probe, synth(n_eval, N_train, k_train, atk, True, st, 12, dev()))
            recS = (posS > torch.quantile(negS, 0.99)).float().mean().item()
            rows.append({"suite": "B", "probe": kind, "strength": st, "k": K,
                         "N_test": N_test, "N_train": N_train,
                         "recall_long": recL, "recall_train_length": recS,
                         "auroc_long": _auroc(posL, negL),
                         "learned_at_train_length": bool(recS >= 0.99)})
            cells.append(f"{recL:.2f} / {recS:.2f}")
            del probe
            clear()
        print(f"  {st:>9.2f}" + "".join(f"{c:>18s}" for c in cells))
    print(NL + "  second number isolates 'never learned it' from 'learned it, then diluted'")
    return rows


def suite_c(quick: bool) -> list[dict]:
    """Benchmark C - distributed / fragmented signal attack.

    The adversary keeps the TOTAL signal budget fixed but splits it across m
    non-contiguous tokens scattered through the context, so each token carries
    total/m. Mean pooling sees an unchanged sum. A hard max sees a peak falling as 1/m.
    This is the attack aimed specifically at the MultiMax reduction, and it is the
    structural trade-off the padding-invariance theorem does not cover.
    """
    print(NL + "=" * 86)
    print("SUITE C - distributed (non-contiguous, energy-split) signal attack")
    print("=" * 86)
    N_train = 512
    N_test = 16384 if not quick else 8192
    total = 2.0                     # total budget == 4 tokens x strength 0.5 from Suite B
    spreads = (1, 4, 16, 64, 256)
    n_eval = 50 if not quick else 25
    n_train = 100 if not quick else 60
    atk = make_attack_direction(D_MODEL)
    kinds = ("multimax", "softmax_attn", "mean_pool")

    print(f"  total signal budget fixed at {total} (= 4 tokens x 0.5); split across m")
    print(f"  non-contiguous tokens, so per-token magnitude is {total}/m")
    print(f"  N_test={N_test}, trained at N={N_train} with the SAME m" + NL)
    print(f"  {'m':>5s}{'per-token':>11s}" + "".join(f"{k:>16s}" for k in kinds))

    rows: list[dict] = []
    for m in spreads:
        cells = []
        for kind in kinds:
            cfg = ProbeConfig(d_model=D_MODEL, hidden=512, n_heads=8, chunk_size=4096)
            probe = train_probe_short(kind, cfg, atk, N_train, m, total,
                                      n_per_class=n_train, epochs=20, seed=0,
                                      contiguous=False, split_energy=True)
            neg = scores_of(probe, synth(n_eval, N_test, 0, atk, False, total, 21, dev(),
                                         contiguous=False, split_energy=True))
            pos = scores_of(probe, synth(n_eval, N_test, m, atk, True, total, 22, dev(),
                                         contiguous=False, split_energy=True))
            rec = (pos > torch.quantile(neg, 0.99)).float().mean().item()
            rows.append({"suite": "C", "probe": kind, "m_spread": m,
                         "per_token_strength": total / m, "total_budget": total,
                         "N_test": N_test, "recall": rec,
                         "auroc": _auroc(pos, neg)})
            cells.append(f"{rec:.3f}")
            del probe
            clear()
        print(f"  {m:>5d}{total/m:>11.4f}" + "".join(f"{c:>16s}" for c in cells))

    print(NL + "  Failure boundary (lowest m at which recall < 0.80):")
    boundary = {}
    for kind in kinds:
        fails = [r["m_spread"] for r in rows if r["probe"] == kind and r["recall"] < 0.80]
        boundary[kind] = min(fails) if fails else None
        print(f"    {kind:<14s} {boundary[kind] if boundary[kind] is not None else 'never'}")
    return rows, boundary


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
    c, c_boundary = suite_c(args.quick)

    # ---- verdicts -------------------------------------------------------------------
    print(NL + "=" * 86)
    print("VERDICTS")
    print("=" * 86)

    alphas = {r["probe"]: r["mem_scaling_alpha"] for r in a if "mem_scaling_alpha" in r}
    print("  A. memory scaling of the aggregator's own overhead (log-log fit, N>=8192):")
    for k_, v_ in alphas.items():
        print(f"       {k_:<14s} alpha = {v_:.3f}")
    mm_a = alphas.get("multimax")
    print(f"     MultiMax O(1) in N            : "
          f"{'CONFIRMED' if mm_a is not None and mm_a < 0.35 else 'NOT CONFIRMED'}")
    oom = [r["probe"] for r in a if r.get("status") == "OOM"]
    print(f"     logged OOM (system limit, not crash): {oom or 'none'}")

    print(NL + "  B. contiguous needle, does MultiMax learn at low signal after annealing?")
    for st in sorted({r["strength"] for r in b}):
        mm = [r for r in b if r["probe"] == "multimax" and r["strength"] == st][0]
        sm = [r for r in b if r["probe"] == "softmax_attn" and r["strength"] == st][0]
        print(f"       strength {st:<5.2f} multimax {mm['recall_long']:.2f}/"
              f"{mm['recall_train_length']:.2f}   softmax {sm['recall_long']:.2f}/"
              f"{sm['recall_train_length']:.2f}")
    low = [r for r in b if r["probe"] == "multimax" and r["strength"] <= 0.10]
    learned_low = all(r["learned_at_train_length"] for r in low) if low else False
    print(f"     annealing fixes the strength-0.10 training failure: "
          f"{'YES' if learned_low else 'NO'}")

    print(NL + "  C. distributed attack failure boundary (lowest m with recall < 0.80):")
    for k_, v_ in c_boundary.items():
        print(f"       {k_:<14s} {v_ if v_ is not None else 'never'}")

    out = {
        "env": env,
        "suite_a_latency_memory": a,
        "suite_b_strength_sweep": b,
        "suite_c_distributed_attack": c,
        "verdicts": {
            "memory_scaling_alpha": alphas,
            "multimax_memory_is_O1_in_N": bool(mm_a is not None and mm_a < 0.35),
            "logged_oom": oom,
            "annealing_fixes_low_signal_training": bool(learned_low),
            "distributed_attack_failure_boundary_m": c_boundary,
        },
    }
    p = ROOT / "benchmark_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  -> wrote {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
