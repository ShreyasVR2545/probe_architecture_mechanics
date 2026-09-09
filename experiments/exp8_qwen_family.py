"""Experiment 8: does the aggregation result hold on a second model family?

exp1 established the ordering on Mistral-7B-v0.1. A single model cannot distinguish "this
is a property of the reduction" from "this is a property of Mistral's residual stream", so
this repeats the comparison on Qwen2.5-7B, a different family (different tokeniser,
pre-training corpus, and hidden width: 28 layers, d=3584 against Mistral's 32 and 4096).

Gemma-7B was the brief's first choice and google/gemma-7b returns HTTP 403 for this
account, as does google/gemma-2-9b, so we use the named alternative.

Layers are chosen by DEPTH RATIO rather than by index, since the two models differ in
depth: 50%, 75% and 100% of 28 layers gives 14, 21 and 27, matching Mistral's 16, 24 and
31 out of 32 in relative position. Comparing layer 16 to layer 16 across models of
different depth would confound depth with family.

Top-r is included alongside MultiMax and softmax pooling because exp4 showed it is the one
countermeasure that survived testing, so it belongs in any generalisation claim.

Split is needle-disjoint, matching the corrected exp1 protocol.

  python experiments/exp8_qwen_family.py [--quick] [--max-n 4096]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimax_probe import ProbeConfig, anneal_tau                           # noqa: E402
from experiments.common import (ART, auroc, clear, dev, make_probe, peak_mib,  # noqa: E402
                                recall_at_fpr, save, timed)
from experiments.exp1_real_residuals import (HARMFUL, BENIGN_NEEDLE, FILLER,   # noqa: E402
                                             build_text)

MODEL_ID = "Qwen/Qwen2.5-7B"
DEPTH_RATIOS = (0.50, 0.75, 1.00)
CACHE = ART / "qwen_acts"
PROBES = ("multimax", "topr", "softmax_attn", "mean_pool")


def layers_for(n_layers: int):
    """Depth ratios -> 0-indexed block indices, clamped to the last block."""
    return tuple(min(int(round(rho * n_layers)) - 1, n_layers - 1) for rho in DEPTH_RATIOS)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  loading {MODEL_ID} (bf16, sharded GPU/CPU) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "5GiB", "cpu": "12GiB"},
        offload_folder=str(ART / "_offload_qwen"), attn_implementation="sdpa",
        low_cpu_mem_usage=True)
    model.eval()
    cfg = model.config
    print(f"  layers={cfg.num_hidden_layers} d={cfg.hidden_size} "
          f"ctx={cfg.max_position_embeddings}")
    return tok, model


@torch.no_grad()
def extract(tok, model, Ns, n_per_class, layers):
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(0)
    for N in Ns:
        for label, positive in ((1, True), (0, False)):
            for i in range(n_per_class):
                f = CACHE / f"N{N}_y{label}_{i:03d}.pt"
                if f.exists():
                    continue
                txt = build_text(i, N, positive, tok, rng)
                ids = tok(txt, return_tensors="pt", truncation=True,
                          max_length=N).input_ids
                if ids.shape[1] < N:
                    need = N - ids.shape[1]
                    extra = tok(FILLER * (need // 50 + 2),
                                return_tensors="pt").input_ids[:, :need]
                    ids = torch.cat([ids, extra], dim=1)
                out = model(input_ids=ids[:, :N], output_hidden_states=True,
                            use_cache=False)
                torch.save({int(l): out.hidden_states[l + 1][0].detach().to(
                    "cpu", torch.float16) for l in layers}, f)
                del out, ids
                clear()
            print(f"    N={N:>6} label={label}: {n_per_class} cached")


def load_split(N, layer, n_per_class):
    pos, neg = [], []
    for label, bucket in ((1, pos), (0, neg)):
        for i in range(n_per_class):
            f = CACHE / f"N{N}_y{label}_{i:03d}.pt"
            if f.exists():
                bucket.append((i, torch.load(f, map_location="cpu")[layer]))
    return pos, neg


def needle_split(items):
    k = len(HARMFUL)
    return ([x for i, x in items if (i % k) < k // 2],
            [x for i, x in items if (i % k) >= k // 2])


def train_eval(kind, d_model, tr_p, tr_n, te_p, te_n, epochs, r=8):
    cfg = ProbeConfig(d_model=d_model, hidden=512, n_heads=8, chunk_size=4096)
    torch.manual_seed(23)
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
        o = []
        for t in items:
            g = t.to(dev(), torch.float32)
            o.append(probe.logits(g).float().reshape(-1)[0])
            del g
        return torch.stack(o)

    sp, sn = sc(te_p), sc(te_n)
    res = {"auroc": auroc(sp, sn), "recall_at_1pct_fpr": recall_at_fpr(sp, sn, 0.01)}
    del probe
    clear()
    return res


def systems(d_model, Ns):
    rows = []
    for N in Ns:
        for kind in PROBES:
            cfg = ProbeConfig(d_model=d_model, hidden=512, n_heads=8, chunk_size=4096)
            probe = make_probe(kind, cfg, r=8).to(dev()).eval().to_compute_dtype()
            try:
                x = torch.randn(N, d_model, device=dev(), dtype=torch.bfloat16)
                with torch.no_grad():
                    _, pk = peak_mib(lambda: probe.logits(x))
                    ms = timed(lambda: probe.logits(x))
                rows.append({"N": N, "probe": kind, "d_model": d_model,
                             "input_mib": x.numel() * 2 / 2 ** 20,
                             "peak_overhead_mib": pk, "latency_ms": ms, "oom": False})
                print(f"  N={N:>7} {kind:<13} overhead={pk:8.1f} MiB  {ms:7.3f} ms")
                del x
            except torch.cuda.OutOfMemoryError:
                rows.append({"N": N, "probe": kind, "oom": True})
                print(f"  N={N:>7} {kind:<13} OOM")
            del probe
            clear()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-n", type=int, default=4096)
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args()
    n_pc = 16 if a.quick else 48
    epochs = 4 if a.quick else 10
    Ns = [n for n in (256, 1024, 4096) if n <= a.max_n]

    d_model, n_layers, ctx = 3584, 28, 131072
    layers = layers_for(n_layers)
    if not a.skip_extract:
        tok, model = load_model()
        d_model = model.config.hidden_size
        n_layers = model.config.num_hidden_layers
        ctx = model.config.max_position_embeddings
        layers = layers_for(n_layers)
        print(f"  depth ratios {DEPTH_RATIOS} -> layers {layers} of {n_layers}")
        extract(tok, model, Ns, n_pc, layers)
        del model, tok
        gc.collect()
        clear()

    print("\n== detection, needle-disjoint split ==")
    det = []
    for N in Ns:
        for li, layer in enumerate(layers):
            pos, neg = load_split(N, layer, n_pc)
            if len(pos) < 8:
                continue
            tr_p, te_p = needle_split(pos)
            tr_n, te_n = needle_split(neg)
            for kind in PROBES:
                res = train_eval(kind, d_model, tr_p, tr_n, te_p, te_n, epochs)
                det.append({"N": N, "layer": layer, "depth_ratio": DEPTH_RATIOS[li],
                            "probe": kind, **res})
                print(f"  N={N:>6} L{layer:<3} ({DEPTH_RATIOS[li]:.0%}) {kind:<13} "
                      f"AUROC={res['auroc']:.3f} rec={res['recall_at_1pct_fpr']:.3f}")
            del pos, neg, tr_p, te_p, tr_n, te_n
            clear()

    print("\n== systems at Qwen width ==")
    sys_ns = [4096, 16_384, 65_536, 131_072] if not a.quick else [4096, 16_384]
    sysrows = systems(d_model, sys_ns)

    def mean_auroc(kind):
        v = [r["auroc"] for r in det if r["probe"] == kind]
        return sum(v) / len(v) if v else None
    means = {k: mean_auroc(k) for k in PROBES}
    ok = [r for r in sysrows if not r.get("oom")]
    mm = [r["peak_overhead_mib"] for r in ok
          if r["probe"] == "multimax" and r["N"] >= 4096]
    verdicts = {
        "model": MODEL_ID, "family": "Qwen2.5", "n_layers": n_layers,
        "d_model": d_model, "native_context": ctx,
        "gemma_gated_403": True,
        "depth_ratios": list(DEPTH_RATIOS), "layers_used": list(layers),
        "mean_auroc": means,
        "multimax_beats_softmax": (means["multimax"] or 0) > (means["softmax_attn"] or 0),
        "multimax_overhead_flat": (max(mm) - min(mm) < 0.01) if len(mm) > 1 else None,
        "multimax_overhead_mib": mm[0] if mm else None,
        "softmax_overhead_at_max_N": next(
            (r["peak_overhead_mib"] for r in ok
             if r["probe"] == "softmax_attn" and r["N"] == max(sys_ns)), None),
    }
    print("\nverdicts:", json.dumps(verdicts, indent=1))
    save("exp8_qwen_family.json",
         {"detection": det, "systems": sysrows, "verdicts": verdicts,
          "config": {"model": MODEL_ID, "n_per_class": n_pc, "epochs": epochs,
                     "Ns": Ns, "quick": a.quick}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
