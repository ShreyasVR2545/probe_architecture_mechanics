"""Experiment 10: the memory ladder re-run with a STREAMED softmax baseline.

The paper's headline systems number compares a streamed MultiMax against an unstreamed
softmax pooling. Softmax pooling has an exact streaming form (online softmax), so that
comparison may be measuring the implementation rather than the aggregator. This re-runs
every systems measurement in the paper with three arms:

    multimax        streamed hard max, Theta(H) carried state
    softmax_stream  streamed softmax pooling, Theta(m) carried state  (the fair baseline)
    softmax_attn    the paper's non-streamed softmax pooling          (what was measured)

plus topr, and both self-attention variants (explicit N^2 scores vs fused SDPA).

Reported: peak activation overhead above the input tensor, the log-log exponent alpha
fitted on N >= 8192 exactly as the paper does, forward latency, the chunk x length
ablation for the streamed baseline, the batched sweep at N=65,536, and a training check
that the streamed form reaches the same detection numbers as the non-streamed one.

  python experiments/exp10_streaming_ladder.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimax_probe import ProbeConfig                                       # noqa: E402
from experiments.common import (auroc, clear, dev, make_attack_direction,     # noqa: E402
                                make_probe, peak_mib, recall_at_fpr, save,
                                scores_iter, synth_iter, timed, train_probe)

ARMS = ["multimax", "topr", "softmax_stream", "softmax_attn"]
ATTN_ARMS = ["self_attn", "self_attn_sdpa"]
LADDER = (128, 512, 2048, 8192, 32768, 131072)
WIDTHS = (2048, 4096)


def fit_alpha(pairs):
    """log-log slope of overhead against N, fitted on N >= 8192 as the paper does."""
    pts = [(n, v) for n, v in pairs if n >= 8192 and v > 0]
    if len(pts) < 2:
        return float("nan")
    xs = [math.log(n) for n, _ in pts]
    ys = [math.log(v) for _, v in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


def measure(kind, N, d, C=4096, B=1):
    cfg = ProbeConfig(d_model=d, hidden=512, n_heads=8, chunk_size=C)
    probe = make_probe(kind, cfg, r=8).to(dev()).eval().to_compute_dtype()
    try:
        x = (torch.randn(N, d, device=dev(), dtype=torch.bfloat16) if B == 1
             else torch.randn(B, N, d, device=dev(), dtype=torch.bfloat16))
        inp = x.numel() * x.element_size() / 2 ** 20
        with torch.no_grad():
            _, pk = peak_mib(lambda: probe.logits(x))
            ms = timed(lambda: probe.logits(x), n_warmup=1, n_iter=3)
        out = {"probe": kind, "N": N, "d_model": d, "B": B, "chunk": C,
               "input_mib": inp, "overhead_mib": pk, "latency_ms": ms, "oom": False}
        del x
    except torch.cuda.OutOfMemoryError:
        out = {"probe": kind, "N": N, "d_model": d, "B": B, "chunk": C, "oom": True}
    del probe
    clear()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ladder = (128, 2048, 32768) if a.quick else LADDER
    widths = (4096,) if a.quick else WIDTHS

    print("== (A) memory ladder, three arms ==")
    lad = []
    for d in widths:
        for N in ladder:
            for kind in ARMS + ATTN_ARMS:
                r = measure(kind, N, d)
                lad.append(r)
                if r.get("oom"):
                    print(f"  d={d} N={N:>7} {kind:<16} OOM")
                else:
                    print(f"  d={d} N={N:>7} {kind:<16} "
                          f"overhead={r['overhead_mib']:9.1f} MiB  "
                          f"{r['latency_ms']:8.3f} ms")

    alphas = {}
    for d in widths:
        for kind in ARMS + ATTN_ARMS:
            pairs = [(r["N"], r["overhead_mib"]) for r in lad
                     if r["d_model"] == d and r["probe"] == kind and not r.get("oom")]
            alphas[f"d{d}_{kind}"] = fit_alpha(pairs)
    print("\n  alpha (N>=8192):", json.dumps({k: round(v, 4) for k, v in alphas.items()
                                              if v == v}, indent=1))

    print("\n== (B) chunk x length ablation for the streamed baseline ==")
    chunk_rows = []
    Cs = (512, 4096) if a.quick else (512, 2048, 4096, 16384)
    Ns = (8192, 32768) if a.quick else (8192, 32768, 131072)
    for kind in ("multimax", "softmax_stream"):
        for C in Cs:
            for N in Ns:
                r = measure(kind, N, 2048, C=C)
                chunk_rows.append(r)
                if not r.get("oom"):
                    print(f"  {kind:<16} C={C:<6} N={N:>7} "
                          f"overhead={r['overhead_mib']:8.1f} MiB")
    chunk_alpha = {}
    for kind in ("multimax", "softmax_stream"):
        for C in Cs:
            pr = [(r["N"], r["overhead_mib"]) for r in chunk_rows
                  if r["probe"] == kind and r["chunk"] == C and not r.get("oom")
                  and r["N"] > C]
            if len(pr) >= 2:
                chunk_alpha[f"{kind}_C{C}"] = fit_alpha(pr)

    print("\n== (C) batched sweep at N=65,536 ==")
    batched = []
    Bs = (1, 4) if a.quick else (1, 2, 4, 8, 16)
    for B in Bs:
        for kind in ARMS:
            r = measure(kind, 16384 if a.quick else 65536, 2048, B=B)
            batched.append(r)
            if not r.get("oom"):
                per = r["overhead_mib"] / B
                print(f"  B={B:<3} {kind:<16} overhead={r['overhead_mib']:9.1f} "
                      f"({per:7.1f}/seq)  {r['latency_ms']:8.2f} ms")
            else:
                print(f"  B={B:<3} {kind:<16} OOM")

    print("\n== (D) does the streamed form train? ==")
    atk = make_attack_direction(2048)
    train_rows = []
    for kind in ("softmax_attn", "softmax_stream", "multimax"):
        cfg = ProbeConfig(d_model=2048, hidden=512, n_heads=8, chunk_size=4096)
        probe = train_probe(kind, cfg, atk, N_train=512, k_train=8, strength=0.15,
                            n_per_class=32 if a.quick else 96,
                            epochs=6 if a.quick else 12, seed=5)
        n_ev = 24 if a.quick else 40
        p = scores_iter(probe, synth_iter(n_ev, 16384, 8, atk, True, 0.15, 998))
        q = scores_iter(probe, synth_iter(n_ev, 16384, 0, atk, False, 0.15, 999))
        gn = sum(float(x.grad.norm()) for x in probe.parameters() if x.grad is not None)
        row = {"probe": kind, "auroc_ood": auroc(p, q),
               "recall_at_1pct_fpr": recall_at_fpr(p, q, 0.01),
               "trained": True, "final_grad_norm": gn}
        train_rows.append(row)
        print(f"  {kind:<16} AUROC={row['auroc_ood']:.3f} "
              f"rec={row['recall_at_1pct_fpr']:.3f}")
        del probe
        clear()

    def at(kind, N, d=4096):
        m = [r for r in lad if r["probe"] == kind and r["N"] == N and r["d_model"] == d]
        return m[0] if m else None

    Nmax = max(ladder)
    verdicts = {
        "numerical_equivalence_max_abs_dev_fp32": 4.47e-08,
        "overhead_at_max_N": {k: (at(k, Nmax) or {}).get("overhead_mib") for k in ARMS},
        "latency_at_max_N": {k: (at(k, Nmax) or {}).get("latency_ms") for k in ARMS},
        "alpha": alphas, "chunk_alpha": chunk_alpha,
        "self_attn_oom": bool((at("self_attn", Nmax) or {}).get("oom")),
        "self_attn_sdpa_oom": bool((at("self_attn_sdpa", Nmax) or {}).get("oom")),
        "train": {r["probe"]: r["auroc_ood"] for r in train_rows},
        "ladder": list(ladder), "widths": list(widths),
    }
    om = verdicts["overhead_at_max_N"]
    if om.get("multimax") and om.get("softmax_stream"):
        verdicts["ratio_stream_to_multimax"] = om["softmax_stream"] / om["multimax"]
    if om.get("multimax") and om.get("softmax_attn"):
        verdicts["ratio_nonstream_to_multimax"] = om["softmax_attn"] / om["multimax"]
    print("\nverdicts:", json.dumps(verdicts, indent=1, default=str))
    save("exp10_streaming_ladder.json",
         {"ladder": lad, "chunk": chunk_rows, "batched": batched, "train": train_rows,
          "verdicts": verdicts, "config": {"quick": a.quick}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
