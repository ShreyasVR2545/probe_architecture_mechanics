"""
Shared infrastructure: config, VRAM accounting, residual-stream hooks, SAE loading.

Design constraints (see README for why these are hard limits on this machine):
  * Physical VRAM is 7.96 GiB (RTX 5070 Laptop). Usable headroom ~6.8 GiB.
  * Batch size 1. Residual-stream capture is last-token-only by default.
  * Captured activations are immediately detached, cast to float32, moved to CPU.
    Nothing accumulates on device across batches.
"""
from __future__ import annotations

import gc
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
for _d in (LOGS, ARTIFACTS, FIGURES, DATA):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    """Single source of truth for model / SAE / memory settings.

    Model choice is forced by two hard external facts, not preference:

    1. Hardware. The card is 7.96 GiB total / ~6.8 GiB usable, not 16 GiB. Both models
       named in the brief are physically too large: Llama-3.1-8B bf16 ~16 GiB,
       gemma-3-4b-it bf16 ~8.6 GiB.
    2. Licence gating. Every google/gemma-* and meta-llama/* repo returns 403
       GatedRepoError on this account, so Gemma Scope is unreachable in practice even
       though the SAE repos themselves are public.

    SmolLM2-1.7B + EleutherAI's layer-17 32x TopK SAE is the pairing that is both
    ungated and dimension-matched (d_in 2048 == hidden_size 2048), with an 8192-token
    native context window -- which is exactly the short -> long OOD axis Phase 2 needs.

    Known caveat, measured rather than assumed (see Phase 1 step 1.7): the SAE was
    trained on the *base* model HuggingFaceTB/SmolLM2-1.7B while probing runs on the
    *instruct* model. Phase 1 reports reconstruction FVU on both so the size of that
    mismatch is a number in the log, not a hope.
    """

    model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    base_model_id: str = "HuggingFaceTB/SmolLM2-1.7B"   # SAE's training distribution
    sae_repo: str = "EleutherAI/sae-SmolLM2-1.7B-layer17-32x"
    sae_kind: str = "topk"                               # {"topk", "jumprelu"}
    sae_layer: int = 17
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"

    # Residual-stream layers to hook (resid_post). Layer 17 is the SAE site; 11 and 21
    # bracket it for the layer-sweep robustness checks.
    hook_layers: tuple[int, ...] = (11, 17, 21)

    # Memory ceiling. NOTE: the brief specified < 11.5 GiB, which exceeds this card's
    # 7.96 GiB total. The operative ceiling is set from measured capacity instead.
    vram_ceiling_gib: float = 7.0

    seed: int = 0


CFG = Config()


def set_seed(seed: int = 0) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# VRAM accounting
# --------------------------------------------------------------------------------------
GIB = 1024 ** 3


def vram() -> dict[str, float]:
    """Memory snapshot. Tolerates a CUDA context left unhealthy by a prior OOM."""
    zero = {"allocated": 0.0, "reserved": 0.0, "peak_reserved": 0.0, "free": 0.0, "total": 0.0}
    if not torch.cuda.is_available():
        return zero
    try:
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated": torch.cuda.memory_allocated() / GIB,
            "reserved": torch.cuda.memory_reserved() / GIB,
            "peak_reserved": torch.cuda.max_memory_reserved() / GIB,
            "free": free / GIB,
            "total": total / GIB,
        }
    except Exception:
        try:
            return zero | {"reserved": torch.cuda.memory_reserved() / GIB,
                           "peak_reserved": torch.cuda.max_memory_reserved() / GIB}
        except Exception:
            return zero


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def release() -> None:
    """Drop cached blocks. Called between context-length stages, not inside loops."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def vram_probe(label: str, ceiling_gib: float | None = None, verbose: bool = True):
    """Measure peak reserved VRAM across a block and assert the ceiling."""
    ceiling = CFG.vram_ceiling_gib if ceiling_gib is None else ceiling_gib
    reset_peak()
    t0 = time.perf_counter()
    rec: dict = {"label": label}
    try:
        yield rec
    finally:
        dt = time.perf_counter() - t0
        m = vram()
        rec.update(seconds=round(dt, 3), peak_reserved_gib=round(m["peak_reserved"], 3),
                   allocated_gib=round(m["allocated"], 3), ceiling_gib=ceiling,
                   within_ceiling=bool(m["peak_reserved"] <= ceiling))
        if verbose:
            flag = "OK " if rec["within_ceiling"] else "OVER"
            print(f"  [{flag}] {label:<44s} peak={rec['peak_reserved_gib']:.3f} GiB  "
                  f"({dt:.2f}s)")


# --------------------------------------------------------------------------------------
# Model loading + residual-stream hooks
# --------------------------------------------------------------------------------------
def load_model(cfg: Config = CFG):
    """Load model + tokenizer in bf16 on GPU. Returns (model, tokenizer, meta)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    # sdpa, not eager: eager materialises the full (B, H, T, T) score matrix and
    # upcasts the softmax to fp32, which is ~2 GiB per layer at T=4096 with 32 heads
    # and OOMs this card during the long-context sweep. We hook decoder-layer outputs,
    # not attention internals, so the fused kernel costs us nothing observationally.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=cfg.dtype, device_map=None, attn_implementation="sdpa",
    )
    model.to(cfg.device)
    model.eval()
    model.requires_grad_(False)

    blocks = get_blocks(model)
    hcfg = model.config
    text_cfg = getattr(hcfg, "text_config", hcfg)
    meta = {
        "model_id": cfg.model_id,
        "n_layers_config": int(getattr(text_cfg, "num_hidden_layers", len(blocks))),
        "n_blocks_found": len(blocks),
        "d_model": int(getattr(text_cfg, "hidden_size")),
        "n_heads": int(getattr(text_cfg, "num_attention_heads", -1)),
        "n_kv_heads": int(getattr(text_cfg, "num_key_value_heads", -1)),
        "head_dim": int(getattr(text_cfg, "head_dim", -1)),
        "dtype": str(cfg.dtype),
        "n_params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 3),
    }
    return model, tok, meta


def get_blocks(model) -> torch.nn.Module:
    """Locate the transformer block list robustly across HF layouts."""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                 "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, (torch.nn.ModuleList, list)) and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not locate transformer blocks on this model.")


class ResidualCapture:
    """Forward hooks on decoder blocks -> residual stream *after* each block.

    This is the `blocks.{i}.hook_resid_post` site that Gemma Scope `resid_post`
    SAEs are trained on.

    mode="last"  : store only the final-token vector           (d,)      [default]
    mode="all"   : store the full sequence                     (T, d)    [Phase 2/3 only]

    Everything is detached, cast to float32 and moved to CPU inside the hook, so
    no activation memory accumulates on device.
    """

    def __init__(self, model, layers: Sequence[int], mode: str = "last"):
        assert mode in ("last", "all")
        self.blocks = get_blocks(model)
        self.layers = list(layers)
        self.mode = mode
        self.acts: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _mk_hook(self, idx: int) -> Callable:
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out      # (B, T, d)
            if self.mode == "last":
                v = h[:, -1, :]
            else:
                v = h
            self.acts[idx] = v.detach().to(torch.float32).cpu()
            return None  # never modify the forward pass
        return hook

    def __enter__(self) -> "ResidualCapture":
        for i in self.layers:
            self._handles.append(self.blocks[i].register_forward_hook(self._mk_hook(i)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def pop(self) -> dict[int, torch.Tensor]:
        out, self.acts = self.acts, {}
        return out


@torch.no_grad()
def forward_capture(model, input_ids: torch.Tensor, layers: Sequence[int],
                    mode: str = "last") -> dict[int, torch.Tensor]:
    """Single forward pass with residual capture. use_cache=False saves KV memory.

    Deliberately torch.no_grad() and not torch.inference_mode(): inference-mode tensors
    are permanently barred from autograd, and these activations are the *training data*
    for the probes in Phase 2. The memory difference is negligible here because the
    captured tensors leave the device immediately.
    """
    with ResidualCapture(model, layers, mode=mode) as cap:
        model(input_ids=input_ids, use_cache=False)
        return cap.pop()


# --------------------------------------------------------------------------------------
# Gemma Scope SAE loading (direct from hub; no sae_lens dependency)
# --------------------------------------------------------------------------------------
class SAE(torch.nn.Module):
    """Residual-stream sparse autoencoder supporting both public formats.

    Convention here (matches EleutherAI `sparsify`):
        W_enc : (d_sae, d_model)   -- encoder.weight, applied as x @ W_enc.T
        W_dec : (d_sae, d_model)   -- applied as f @ W_dec
        pre   = (x - b_dec) @ W_enc.T + b_enc
        f     = TopK(pre, k)                       [kind="topk",     EleutherAI]
              = ReLU(pre) * 1[pre > threshold]     [kind="jumprelu", Gemma Scope]
        x_hat = f @ W_dec + b_dec

    Held on CPU in float32; `.to(device)` only when a batch is being encoded. The
    decoder rows W_dec[i] are the *feature directions in residual space* -- these are
    what probe weight vectors get projected onto in Phase 3.
    """

    def __init__(self, d_model: int, d_sae: int, kind: str = "topk", k: int = 32):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
        self.threshold = torch.nn.Parameter(torch.zeros(d_sae))
        self.d_model, self.d_sae, self.kind, self.k = d_model, d_sae, kind, k

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc.T + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.pre_acts(x)
        if self.kind == "topk":
            vals, idx = torch.topk(pre, self.k, dim=-1)
            out = torch.zeros_like(pre)
            out.scatter_(-1, idx, torch.relu(vals))
            return out
        return torch.relu(pre) * (pre > self.threshold)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @property
    def decoder_directions(self) -> torch.Tensor:
        """(d_sae, d_model), L2-normalised. The basis Phase 3 projects probes onto."""
        return self.W_dec / self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def load_sae(layer: int | None = None, cfg: Config = CFG) -> tuple[SAE, dict]:
    """Download and load the configured SAE from the hub (no sae_lens dependency)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    layer = cfg.sae_layer if layer is None else layer
    sae_cfg: dict = {}
    for fname in ("cfg.json", "config.json"):
        try:
            sae_cfg = json.loads(Path(hf_hub_download(cfg.sae_repo, filename=fname)).read_text())
            break
        except Exception:
            continue

    raw = None
    for fname in ("sae.safetensors", "params.safetensors"):
        try:
            raw = load_file(hf_hub_download(cfg.sae_repo, filename=fname))
            break
        except Exception:
            continue
    if raw is None:
        raise FileNotFoundError(f"no SAE weight file found in {cfg.sae_repo}")

    key = {k.lower().replace(".", "_").lstrip("_"): k for k in raw}

    def g(*names, required: bool = True):
        for n in names:
            if n in key:
                return raw[key[n]].to(torch.float32)
        if required:
            raise KeyError(f"none of {names} in {list(raw)}")
        return None

    W_dec = g("w_dec", "decoder_weight")
    W_enc = g("encoder_weight", "w_enc")
    if W_enc.shape != W_dec.shape:          # Gemma Scope stores W_enc as (d_model, d_sae)
        W_enc = W_enc.T.contiguous()
    d_sae, d_model = W_dec.shape

    k = int(sae_cfg.get("k", 32))
    sae = SAE(d_model, d_sae, kind=cfg.sae_kind, k=k)
    with torch.no_grad():
        sae.W_enc.copy_(W_enc)
        sae.W_dec.copy_(W_dec)
        b_enc = g("encoder_bias", "b_enc", required=False)
        sae.b_enc.copy_(b_enc if b_enc is not None else torch.zeros(d_sae))
        b_dec = g("b_dec", required=False)
        sae.b_dec.copy_(b_dec if b_dec is not None else torch.zeros(d_model))
        thr = g("threshold", required=False)
        sae.threshold.copy_(thr if thr is not None else torch.zeros(d_sae))
    sae.eval().requires_grad_(False)

    meta = {"layer": layer, "repo": cfg.sae_repo, "kind": cfg.sae_kind, "k": k,
            "d_model": d_model, "d_sae": d_sae,
            "params_mib": round(sum(p.numel() for p in sae.parameters()) * 4 / 1024**2, 1),
            "trained_on": sae_cfg.get("model") or "(see train_config.json)",
            "config": sae_cfg}
    return sae, meta


# --------------------------------------------------------------------------------------
# Null distribution utilities (the 1/sqrt(d) floor)
# --------------------------------------------------------------------------------------
def random_unit_vectors(n: int, d: int, seed: int = 0, device="cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(n, d, generator=g, dtype=torch.float32)
    return (v / v.norm(dim=-1, keepdim=True)).to(device)


def cosine_null(d: int, n_pairs: int = 5000, seed: int = 0) -> dict[str, float]:
    """Empirical distribution of |cos| between independent random unit vectors in R^d.

    The analytic scale is 1/sqrt(d); this returns the measured quantiles so that any
    observed cosine can be reported as a percentile rather than a raw number.
    """
    a = random_unit_vectors(n_pairs, d, seed=seed)
    b = random_unit_vectors(n_pairs, d, seed=seed + 12345)
    c = (a * b).sum(-1).abs()
    q = torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999])
    qs = torch.quantile(c, q).tolist()
    return {"d": d, "n_pairs": n_pairs, "analytic_1_over_sqrt_d": 1.0 / d ** 0.5,
            "mean_abs_cos": c.mean().item(), "std_abs_cos": c.std().item(),
            "p50": qs[0], "p90": qs[1], "p95": qs[2], "p99": qs[3], "p999": qs[4],
            "max": c.max().item()}


def cosine_percentile(observed: float, null: dict) -> float:
    """Where an observed |cos| sits in the null, via the Gaussian approximation."""
    from math import erf, sqrt
    z = (abs(observed) - null["mean_abs_cos"]) / max(null["std_abs_cos"], 1e-12)
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
def write_json(name: str, payload: dict) -> Path:
    p = LOGS / name
    p.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  -> wrote {p.relative_to(ROOT)}")
    return p


def banner(msg: str) -> None:
    print(f"\n{'=' * 78}\n{msg}\n{'=' * 78}")
