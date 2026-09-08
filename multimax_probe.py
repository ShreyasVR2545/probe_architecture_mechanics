r"""
multimax_probe.py — dilution-resistant activation probes for long-context monitoring.

Production module. No training-time dependencies, no dataset dependencies; operates on
hidden-state tensors handed to it by a forward hook.

--------------------------------------------------------------------------------------
MODEL
--------------------------------------------------------------------------------------
For a sequence of hidden activations x_{i,j} in R^d, j = 1..N:

    y_{i,j} = MLP(x_{i,j})                          token-level transform
    a_h     = max_{1<=j<=N} ( v_h^T y_{i,j} )       hard-max reduction, head h
    logit_i = sum_{h=1}^{H} a_h + b                 scalar logit
    p_i     = sigmoid( clamp(logit_i, -10, +10) )

The reduction is a hard max over positions, not a softmax-weighted average. That single
choice is what makes the probe invariant to context padding: appending benign tokens can
only add candidates to the max, it cannot down-weight the incumbent. See
`math_formulation.tex` §2 for the dilution proof.

--------------------------------------------------------------------------------------
ENGINEERING GUARANTEES
--------------------------------------------------------------------------------------
1. Numerical safety. Raw logits are clamped to [-10, +10] before the sigmoid. At the
   clamp boundary sigmoid' ~= 4.5e-5, which is small but strictly non-zero, so a
   saturated probe still receives gradient instead of becoming a dead unit. Without the
   clamp, |logit| grows with H and with sequence length (more positions = higher max),
   and fp16 sigmoid saturates to exactly 0 or 1, giving zero gradient and NaN BCE.

2. Mixed precision by stage. The MLP and head projections run in bf16/fp16 (they are
   the FLOP-heavy part and are numerically benign). The max-reduction, the head sum and
   the logit arithmetic are forced to float32: a max over 10^5 positions followed by a
   sum over H heads is exactly where fp16's 5-bit exponent underflows.

3. O(1) activation memory in N. `chunk_size` streams the sequence and keeps only a
   running per-head max of shape (B, H). Peak activation memory is set by the chunk, not
   by N, so a 131,072-token sequence costs the same as a 4,096-token one.

Baselines for comparison are included so that claims about MultiMax are measured against
something rather than asserted:
  * SoftmaxAttnProbe  -- single learned query, softmax over positions.  O(N) memory.
  * SelfAttnProbe     -- full self-attention over the sequence.         O(N^2) memory.
  * MeanPoolProbe     -- uniform average over positions.                O(N) memory.

NOTE ON THE O(N^2) CLAIM. Single-query attention pooling is *already* O(N): the score
tensor is (B, N), not (B, N, N). The quadratic term only appears if the probe runs
self-attention over the sequence. Both are provided so the memory comparison reflects
what is actually being compared.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ProbeConfig",
    "MultiMaxProbe",
    "SoftmaxAttnProbe",
    "SelfAttnProbe",
    "MeanPoolProbe",
    "build_probe",
    "LOGIT_CLAMP",
]

LOGIT_CLAMP: float = 10.0


# ======================================================================================
# Straight-through clamp
# ======================================================================================
class STEClamp(torch.autograd.Function):
    """Hard clamp forward, identity backward.

    torch.clamp has d/dx = 1[|x| < c], i.e. identically zero on the saturated region, so
    using it as a numerical guard produces exactly the zero-gradient trapping the guard
    exists to prevent. Measured on a saturated MultiMax logit: grad norm 0.000e+00.

    Forward is the exact clamp the spec requires; backward passes gradient through
    unchanged so a saturated probe keeps training.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.clone(), None, None


def ste_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return STEClamp.apply(x, lo, hi)


# ======================================================================================
# Config
# ======================================================================================
@dataclass
class ProbeConfig:
    d_model: int
    hidden: int = 512
    n_heads: int = 8
    dropout: float = 0.0
    compute_dtype: torch.dtype = torch.bfloat16   # MLP / projection dtype
    reduce_dtype: torch.dtype = torch.float32     # max / sum / logit dtype
    chunk_size: int = 4096                        # 0 disables chunking
    logit_clamp: float = LOGIT_CLAMP
    straight_through_clamp: bool = True           # see _ProbeBase._clamp
    # Smooth-max temperature. tau = 0 is the hard max of the paper model; tau > 0 is the
    # LogSumExp relaxation used during early training. See MultiMaxProbe.head_activations
    # and anneal_tau().
    tau: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["compute_dtype"] = str(self.compute_dtype)
        d["reduce_dtype"] = str(self.reduce_dtype)
        return d


def _as_btd(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Accept (N, d) or (B, N, d); return (B, N, d) and whether a batch dim was added."""
    if x.dim() == 2:
        return x.unsqueeze(0), True
    if x.dim() == 3:
        return x, False
    raise ValueError(f"expected (N, d) or (B, N, d), got shape {tuple(x.shape)}")


class _ProbeBase(nn.Module):
    """Common plumbing: dtype discipline, clamping, and the predict/logit contract."""

    cfg: ProbeConfig

    def logits(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.logits(x, attention_mask)

    def to_compute_dtype(self) -> "_ProbeBase":
        """Cast the FLOP-heavy projections to cfg.compute_dtype for inference.

        Deliberately opt-in rather than automatic: master weights stay float32 so the
        same module can be trained, and only a deployed probe pays the precision cost.
        The reduction path is unaffected -- it upcasts to reduce_dtype regardless.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.to(self.cfg.compute_dtype)
        return self

    @property
    def _wdtype(self) -> torch.dtype:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                return m.weight.dtype
        return self.cfg.reduce_dtype

    def _clamp(self, logit: torch.Tensor) -> torch.Tensor:
        """Bound the logit to +-clamp WITHOUT creating a zero-gradient trap.

        A plain torch.clamp is the obvious implementation and it is wrong for this
        purpose: its gradient is exactly zero outside the interval, so the guard that
        was meant to prevent zero-gradient trapping causes it. Verified in the
        self-test -- the naive version returns grad norm 0.0 on a saturated logit.

        Straight-through instead: the forward value is exactly the hard clamp the spec
        asks for, while the backward pass sees the identity, so a saturated probe keeps
        training. Set cfg.straight_through_clamp=False for the plain clamp.
        """
        c = self.cfg.logit_clamp
        z = logit.to(self.cfg.reduce_dtype)
        if not self.cfg.straight_through_clamp:
            return z.clamp(-c, c)
        return ste_clamp(z, -c, c)

    def raw_logit(self, x: torch.Tensor,
                  attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pre-clamp logit. Diagnostic only -- shows dilution that the clamp hides."""
        prev = self.cfg.logit_clamp
        self.cfg.logit_clamp = float("inf")
        try:
            return self.logits(x, attention_mask)
        finally:
            self.cfg.logit_clamp = prev

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor,
                      attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """P(harmful). Always float32; always finite; always in (0, 1)."""
        return torch.sigmoid(self.logits(x, attention_mask).float())

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ======================================================================================
# MultiMax
# ======================================================================================
class MultiMaxProbe(_ProbeBase):
    """Multi-head hard-max probe. Dilution-resistant by construction."""

    kind = "multimax"

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
        )
        # v_h stacked: (hidden, H). One column per head.
        self.heads = nn.Linear(cfg.hidden, cfg.n_heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self._init()

    def _init(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.heads.weight, std=self.cfg.hidden ** -0.5)

    # -- internals ---------------------------------------------------------------------
    def _head_scores(self, xc: torch.Tensor) -> torch.Tensor:
        """(B, n, d) -> (B, n, H) head scores v_h^T MLP(x). Computed in compute_dtype."""
        y = self.mlp(xc.to(self._wdtype))
        return self.heads(y)

    def head_activations(self, x: torch.Tensor,
                         attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """a_h = max_j v_h^T y_j, shape (B, H), in reduce_dtype.

        Chunked so peak activation memory depends on `chunk_size`, not on N. The running
        max is carried in float32 -- accumulating a max in fp16 over 10^5 candidates is
        where precision is actually lost.
        """
        xb, _ = _as_btd(x)
        B, N, _ = xb.shape
        rd = self.cfg.reduce_dtype
        chunk = self.cfg.chunk_size or N
        tau = float(self.cfg.tau)
        H = self.cfg.n_heads
        neg_inf = float("-inf")

        # Streaming reduction. tau = 0 is a running max. tau > 0 is the NORMALISED
        # smooth max (Boltzmann operator / log-mean-exp),
        #     smax_tau(s) = tau * log( (1/N) sum_j exp(s_j / tau) )
        #                 = m + tau * log( (1/N) sum_j exp((s_j - m)/tau) ),
        # which -> max_j s_j as tau -> 0 and -> mean_j s_j as tau -> infinity.
        #
        # The 1/N normalisation is essential and its absence was a real bug. Plain
        # LogSumExp adds tau*log(N) per head, i.e. +42 to the logit at tau=1, N=512 and
        # +68 at N=16384 (measured). That offset is both large and LENGTH-DEPENDENT, so
        # an annealing probe has to re-learn its bias every epoch AND sees a different
        # offset at deployment length than at training length. Symptom: Suite B recall
        # was non-monotone in signal strength (1.00 at 0.15, 0.04 at 0.50).
        #
        # Both branches keep O(1) activation memory in N: only (B, H) state is carried.
        m = torch.full((B, H), neg_inf, device=xb.device, dtype=rd)
        Z = torch.zeros((B, H), device=xb.device, dtype=rd) if tau > 0 else None
        cnt = 0

        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            sc = self._head_scores(xb[:, s:e]).to(rd)             # (B, n, H)
            if attention_mask is not None:
                msk = attention_mask[:, s:e].to(torch.bool).unsqueeze(-1)
                sc = sc.masked_fill(~msk, neg_inf)
            cm = sc.amax(dim=1)                                    # (B, H)
            if tau <= 0:
                m = torch.maximum(m, cm)
            else:
                new_m = torch.maximum(m, cm)
                safe = torch.isfinite(new_m)
                shift = torch.where(safe, new_m, torch.zeros_like(new_m))
                # rescale the old accumulator to the new max, then fold in this chunk
                Z = Z * torch.exp((torch.where(torch.isfinite(m), m, shift) - shift) / tau) \
                    * torch.isfinite(m).to(rd)
                Z = Z + torch.exp((sc - shift.unsqueeze(1)) / tau).sum(dim=1)
                m = new_m
                cnt += (e - s)
            del sc

        if tau > 0:
            out = m + tau * torch.log((Z / max(cnt, 1)).clamp_min(1e-30))
        else:
            out = m
        # A fully-masked sequence leaves -inf; fall back to 0 so the logit stays finite.
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

    def set_tau(self, tau: float) -> None:
        """Set the smooth-max temperature (0 = hard max)."""
        self.cfg.tau = max(float(tau), 0.0)

    # -- public ------------------------------------------------------------------------
    def logits(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        _, squeezed = _as_btd(x)
        a = self.head_activations(x, attention_mask)              # (B, H) fp32
        logit = a.sum(dim=-1) + self.bias.to(a.dtype)             # (B,)
        logit = self._clamp(logit)
        return logit.squeeze(0) if squeezed else logit

    @torch.no_grad()
    def argmax_positions(self, x: torch.Tensor,
                         attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, H) index of the token each head selected. The probe's own explanation of
        *where* it fired -- directly usable as an audit pointer in a control protocol."""
        xb, _ = _as_btd(x)
        B, N, _ = xb.shape
        rd = self.cfg.reduce_dtype
        chunk = self.cfg.chunk_size or N
        best = torch.full((B, self.cfg.n_heads), float("-inf"), device=xb.device, dtype=rd)
        idx = torch.zeros((B, self.cfg.n_heads), device=xb.device, dtype=torch.long)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            sc = self._head_scores(xb[:, s:e]).to(rd)
            if attention_mask is not None:
                m = attention_mask[:, s:e].to(torch.bool).unsqueeze(-1)
                sc = sc.masked_fill(~m, float("-inf"))
            v, j = sc.max(dim=1)
            upd = v > best
            idx = torch.where(upd, j + s, idx)
            best = torch.maximum(best, v)
        return idx


# ======================================================================================
# Baselines
# ======================================================================================
class SoftmaxAttnProbe(_ProbeBase):
    """Single learned query, softmax over positions. O(N) memory -- scores are (B, N)."""

    kind = "softmax_attn"

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.hidden), nn.GELU())
        self.query = nn.Parameter(torch.randn(cfg.hidden) * cfg.hidden ** -0.5)
        self.out = nn.Linear(cfg.hidden, 1)

    def logits(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        xb, squeezed = _as_btd(x)
        rd = self.cfg.reduce_dtype
        y = self.mlp(xb.to(self._wdtype))                          # (B, N, h)
        s = (y.to(rd) @ self.query.to(rd)) / self.cfg.hidden ** 0.5
        if attention_mask is not None:
            s = s.masked_fill(~attention_mask.to(torch.bool), float("-inf"))
        w = torch.softmax(s, dim=-1)                               # (B, N)  <- the dilution
        pooled = (y.to(rd) * w.unsqueeze(-1)).sum(dim=1)           # (B, h)
        logit = self.out(pooled.to(self.out.weight.dtype)).squeeze(-1)
        logit = self._clamp(logit)
        return logit.squeeze(0) if squeezed else logit


class SelfAttnProbe(_ProbeBase):
    """Full self-attention over the sequence, then mean-pool. O(N^2) score memory.

    Included because this -- not single-query pooling -- is the architecture for which
    the quadratic-overhead claim is true. It is expected to OOM at long N, and the
    benchmark records where.
    """

    kind = "self_attn"

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.d_model, cfg.hidden)
        self.qkv = nn.Linear(cfg.hidden, 3 * cfg.hidden, bias=False)
        self.out = nn.Linear(cfg.hidden, 1)

    def logits(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        xb, squeezed = _as_btd(x)
        rd, cd = self.cfg.reduce_dtype, self._wdtype
        h = self.proj(xb.to(cd))
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)) / self.cfg.hidden ** 0.5   # (B, N, N)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.to(torch.bool).unsqueeze(1), float("-inf"))
        ctx = torch.softmax(scores.float(), dim=-1).to(cd) @ v
        pooled = ctx.to(rd).mean(dim=1)
        logit = self.out(pooled.to(self.out.weight.dtype)).squeeze(-1)
        logit = self._clamp(logit)
        return logit.squeeze(0) if squeezed else logit


class MeanPoolProbe(_ProbeBase):
    """Uniform average over positions. The canonical dilution-prone aggregator."""

    kind = "mean_pool"

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.hidden), nn.GELU())
        self.out = nn.Linear(cfg.hidden, 1)

    def logits(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        xb, squeezed = _as_btd(x)
        rd = self.cfg.reduce_dtype
        chunk = self.cfg.chunk_size or xb.shape[1]
        acc = torch.zeros(xb.shape[0], self.cfg.hidden, device=xb.device, dtype=rd)
        cnt = 0
        for s in range(0, xb.shape[1], chunk):
            e = min(s + chunk, xb.shape[1])
            y = self.mlp(xb[:, s:e].to(self._wdtype)).to(rd)
            if attention_mask is not None:
                m = attention_mask[:, s:e].to(rd).unsqueeze(-1)
                acc += (y * m).sum(1); cnt += int(m.sum().item())
            else:
                acc += y.sum(1); cnt += (e - s)
            del y
        pooled = acc / max(cnt, 1)
        logit = self.out(pooled.to(self.out.weight.dtype)).squeeze(-1)
        logit = self._clamp(logit)
        return logit.squeeze(0) if squeezed else logit


_REGISTRY = {
    "multimax": MultiMaxProbe,
    "softmax_attn": SoftmaxAttnProbe,
    "self_attn": SelfAttnProbe,
    "mean_pool": MeanPoolProbe,
}


def build_probe(kind: Literal["multimax", "softmax_attn", "self_attn", "mean_pool"],
                cfg: ProbeConfig) -> _ProbeBase:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown probe kind {kind!r}; options: {list(_REGISTRY)}")
    return _REGISTRY[kind](cfg)


def anneal_tau(epoch: int, n_epochs: int, tau_start: float = 1.0,
               frac_anneal: float = 0.6) -> float:
    """Geometric anneal from `tau_start` to hard max (tau = 0).

    Why this exists. The hard-max subgradient is supported on ONE position per head
    (d max_j / d y_j = 1[j = argmax]), so a MultiMax probe sees H tokens of learning
    signal per sequence versus N for softmax pooling. Measured consequence: at attack
    strength 0.10 the probe failed to learn even at its TRAINING length (recall 0.00),
    while softmax reached 1.00 there -- a pure optimisation failure, unrelated to
    dilution, and invisible to the forward-only padding-invariance analysis.

    The LogSumExp relaxation spreads gradient over all positions with softmax weights,
    so early epochs get a dense signal; tau then decays to 0 so the deployed forward map
    is exactly the hard max whose padding invariance is the point of the architecture.

    Anneal completes at `frac_anneal` of training, leaving the remaining epochs to
    fine-tune under the true hard-max objective.
    """
    if n_epochs <= 1:
        return 0.0
    cutoff = max(1, int(frac_anneal * n_epochs))
    if epoch >= cutoff:
        return 0.0
    # geometric decay tau_start -> ~1e-2 over `cutoff` epochs, then snap to 0
    frac = epoch / cutoff
    return float(tau_start * (1e-2 / tau_start) ** frac)


# ======================================================================================
# Self-test
# ======================================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ProbeConfig(d_model=2048, hidden=512, n_heads=8, chunk_size=4096)
    print(f"device={dev}  cfg={cfg.to_dict()}\n")

    p = build_probe("multimax", cfg).to(dev).to_compute_dtype()
    print(f"MultiMaxProbe params: {p.n_params():,}  weight dtype: {p._wdtype}")

    # 1. shape contract
    for shape in [(256, 2048), (4, 256, 2048)]:
        out = p.logits(torch.randn(*shape, device=dev))
        print(f"  input {str(shape):<18s} -> logits {tuple(out.shape)}")

    # 2. chunking invariance: chunked and unchunked must agree
    x = torch.randn(2, 9000, 2048, device=dev)
    a = p.logits(x)
    p.cfg.chunk_size = 0
    b = p.logits(x)
    p.cfg.chunk_size = 4096
    print(f"\n  chunked vs unchunked max |delta| = {(a - b).abs().max().item():.3e}")

    # 3. clamp + gradient survival
    big = torch.randn(1, 128, 2048, device=dev) * 500
    lg = p.logits(big)
    print(f"  extreme input -> logit {lg.item():+.4f} (clamped to +-{cfg.logit_clamp})")
    pr = p.predict_proba(big)
    print(f"  predict_proba = {pr.item():.6f}  finite={bool(torch.isfinite(pr).all())}")
    # Gradient must survive the clamp. Use target=0 against a saturated positive logit:
    # dBCE/dlogit = sigmoid(logit) - target, which is ~1 here. (An earlier version of
    # this test scaled the logit by 1e3 with target=1, which saturates the BCE itself and
    # reports DEAD for both modes -- it was measuring the loss, not the clamp.)
    def grad_norm(straight: bool) -> float:
        p.cfg.straight_through_clamp = straight
        xg = torch.randn(1, 512, 2048, device=dev, requires_grad=True) * 8.0
        xg.retain_grad()
        F.binary_cross_entropy_with_logits(
            p.logits(xg), torch.zeros(1, device=dev)).backward()
        return xg.grad.norm().item()

    g_st, g_hard = grad_norm(True), grad_norm(False)
    p.cfg.straight_through_clamp = True
    print(f"  grad norm, straight-through clamp = {g_st:.3e} "
          f"({'alive' if g_st > 0 else 'DEAD'})")
    print(f"  grad norm, plain torch.clamp      = {g_hard:.3e} "
          f"({'alive' if g_hard > 0 else 'DEAD  <- the trap the flag avoids'})")

    # 3b. STE clamp must keep PARAMETER gradients alive at production sequence lengths
    print("\n  STE clamp: parameter-gradient survival at long N (spec: N >= 32768)")
    for N in (32768, 131072):
        p.zero_grad(set_to_none=True)
        xl = torch.randn(1, N, 2048, device=dev, dtype=torch.bfloat16) * 4.0
        F.binary_cross_entropy_with_logits(p.logits(xl).reshape(1),
                                           torch.zeros(1, device=dev)).backward()
        gsum = sum(q.grad.abs().sum().item() for q in p.parameters() if q.grad is not None)
        raw = p.raw_logit(xl).item()
        assert gsum > 0, f"parameter gradients dead at N={N}"
        print(f"    N={N:>7d}  raw logit {raw:>8.2f} (clamped to {cfg.logit_clamp})  "
              f"sum|param.grad| = {gsum:.4e}  [alive]")
        del xl
        torch.cuda.empty_cache()
    p.zero_grad(set_to_none=True)

    # 3c. Smooth-max must converge to the hard max as tau -> 0
    print("\n  Smooth-max annealing: LSE_tau -> hard max as tau -> 0")
    xs = torch.randn(2, 4096, 2048, device=dev)
    p.set_tau(0.0)
    hard = p.raw_logit(xs)
    for tau in (1.0, 0.3, 0.1, 0.01):
        p.set_tau(tau)
        d = (p.raw_logit(xs) - hard).abs().max().item()
        print(f"    tau={tau:<5.2f}  max|LSE - hardmax| = {d:.4f}")
    p.set_tau(0.0)
    from multimax_probe import anneal_tau
    sched = [round(anneal_tau(e, 20), 4) for e in range(0, 20, 3)]
    print(f"    anneal schedule (20 epochs, sampled): {sched}")

    # gradient density: the reason the schedule exists
    print("\n  Gradient density (fraction of positions receiving signal):")
    for tau in (0.0, 0.5):
        p.set_tau(tau)
        xg3 = torch.randn(1, 2048, 2048, device=dev, requires_grad=True)
        p.logits(xg3).sum().backward()
        frac = (xg3.grad.abs().sum(-1) > 0).float().mean().item()
        print(f"    tau={tau:<5.2f}  {frac*100:6.2f}% of positions  "
              f"({'hard max: H heads only' if tau == 0 else 'LSE: dense'})")
    p.set_tau(0.0)

    # 4. dilution: identical needle, growing benign context
    print("\n  Dilution check (same needle, growing benign padding):")
    torch.manual_seed(1)
    needle = torch.randn(1, 8, 2048, device=dev) * 6.0
    print("  (raw pre-clamp logits, so dilution is visible below saturation)")
    print(f"  {'N':>8s}  {'multimax':>10s}  {'softmax_attn':>13s}  {'mean_pool':>11s}")
    probes = {k: build_probe(k, cfg).to(dev).to_compute_dtype()
              for k in ("multimax", "softmax_attn", "mean_pool")}
    for N in (64, 512, 4096, 32768):
        pad = torch.randn(1, N, 2048, device=dev) * 0.5
        seq = torch.cat([pad[:, :N // 2], needle, pad[:, N // 2:]], dim=1)
        row = {k: v.raw_logit(seq).item() for k, v in probes.items()}
        print(f"  {seq.shape[1]:>8d}  {row['multimax']:>10.4f}  "
              f"{row['softmax_attn']:>13.4f}  {row['mean_pool']:>11.4f}")

    # 5. argmax attribution
    idx = probes["multimax"].argmax_positions(
        torch.cat([torch.randn(1, 2000, 2048, device=dev) * 0.5, needle], dim=1))
    print(f"\n  argmax_positions (needle at 2000-2007): {idx.tolist()[0]}")
    print("\nself-test OK")
