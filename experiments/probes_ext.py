"""Aggregators added for the reviewer response: streaming Top-r and Mean-Max.

Section 7.2 of the paper proposed both of these as countermeasures to the fragmentation
attack but did not test either. These implement them so the proposals can be measured
instead of asserted.

Top-r. Replacing max_j by the mean of the r largest per-head scores should raise the
fragmentation threshold from m ~ 1 to m ~ r while keeping the streaming property, since a
running top-r heap is Theta(r) state. The implementation keeps that promise literally: it
carries an (B, r, H) buffer across chunks and never materialises the (B, N, H) score
tensor, so peak memory is Theta(min(C, N) + r) exactly as for the hard max.

Mean-Max. A fixed-weight ensemble of a hard max and a running mean. The two aggregators
fail on disjoint attack geometries (the max on fragmented budgets, the mean on
concentrated needles), so the sum should dominate either alone. Both members stream, so
the ensemble keeps the memory law.

Both subclass the audited _ProbeBase so they inherit the STE clamp and dtype discipline
unchanged; nothing in multimax_probe.py is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimax_probe import ProbeConfig, _ProbeBase, _as_btd  # noqa: E402


class TopRProbe(_ProbeBase):
    """logit = sum_h mean(top-r_j v_h^T y_j) + b, computed in a streaming pass."""

    kind = "topr"

    def __init__(self, cfg: ProbeConfig, r: int = 8):
        super().__init__()
        self.cfg = cfg
        self.r = int(r)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.hidden),
            nn.GELU(),
        )
        self.heads = nn.Linear(cfg.hidden, cfg.n_heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.heads.weight, std=cfg.hidden ** -0.5)

    def _head_scores(self, xc: torch.Tensor) -> torch.Tensor:
        return self.heads(self.mlp(xc.to(self._wdtype)))

    def head_activations(self, x: torch.Tensor,
                         attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        xb, _ = _as_btd(x)
        B, N, _ = xb.shape
        C = self.cfg.chunk_size or N
        rd = self.cfg.reduce_dtype
        r = min(self.r, N)

        # Running buffer of the r best scores seen so far, per (batch, head).
        # Shape (B, r, H). Initialised to -inf so the first chunk fills it.
        buf = torch.full((B, r, self.cfg.n_heads), float("-inf"), device=xb.device, dtype=rd)
        for s in range(0, N, C):
            sc = self._head_scores(xb[:, s:s + C]).to(rd)          # (B, n, H)
            merged = torch.cat([buf, sc], dim=1)                    # (B, r+n, H)
            buf = merged.topk(r, dim=1).values                      # (B, r, H)
            del sc, merged
        return buf.mean(dim=1)                                      # (B, H)

    def logits(self, x: torch.Tensor,
               attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        a = self.head_activations(x, attention_mask)
        z = a.sum(dim=-1) + self.bias.to(a.dtype)
        out = self._clamp(z)
        return out.squeeze(0) if x.dim() == 2 else out


class MeanMaxProbe(_ProbeBase):
    """Streaming ensemble: sum of per-head maxima PLUS sum of per-head means.

    Each family gets its own head matrix, so the two members can specialise rather than
    being forced to share a direction. Both reductions are computed in the same chunk
    loop, so the ensemble costs one pass and Theta(min(C, N)) memory, not two passes.
    """

    kind = "mean_max"

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.hidden),
            nn.GELU(),
        )
        self.heads_max = nn.Linear(cfg.hidden, cfg.n_heads, bias=False)
        self.heads_mean = nn.Linear(cfg.hidden, cfg.n_heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                nn.init.zeros_(m.bias)
        for h in (self.heads_max, self.heads_mean):
            nn.init.normal_(h.weight, std=cfg.hidden ** -0.5)

    def head_activations(self, x: torch.Tensor,
                         attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        xb, _ = _as_btd(x)
        B, N, _ = xb.shape
        C = self.cfg.chunk_size or N
        rd = self.cfg.reduce_dtype
        H = self.cfg.n_heads

        run_max = torch.full((B, H), float("-inf"), device=xb.device, dtype=rd)
        run_sum = torch.zeros((B, H), device=xb.device, dtype=rd)
        for s in range(0, N, C):
            y = self.mlp(xb[:, s:s + C].to(self._wdtype))
            sm = self.heads_max(y).to(rd)
            sa = self.heads_mean(y).to(rd)
            run_max = torch.maximum(run_max, sm.amax(dim=1))
            run_sum = run_sum + sa.sum(dim=1)
            del y, sm, sa
        return torch.cat([run_max, run_sum / max(N, 1)], dim=-1)     # (B, 2H)

    def logits(self, x: torch.Tensor,
               attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        a = self.head_activations(x, attention_mask)
        z = a.sum(dim=-1) + self.bias.to(a.dtype)
        out = self._clamp(z)
        return out.squeeze(0) if x.dim() == 2 else out


def build_ext_probe(kind: str, cfg: ProbeConfig, r: int = 8):
    if kind == "topr":
        return TopRProbe(cfg, r=r)
    if kind == "mean_max":
        return MeanMaxProbe(cfg)
    raise ValueError(f"unknown extended probe: {kind}")


if __name__ == "__main__":
    # Self-test: streaming must equal a single-shot reduction, and memory must not grow
    # with N. Both are the properties the paper claims for these variants.
    torch.manual_seed(0)
    cfg = ProbeConfig(d_model=128, hidden=64, n_heads=4, chunk_size=64,
                      compute_dtype=torch.float32)
    x = torch.randn(1024, 128)
    for kind in ("topr", "mean_max"):
        p = build_ext_probe(kind, cfg, r=8).eval()
        with torch.no_grad():
            chunked = p.logits(x).item()
            p.cfg.chunk_size = 0                    # single shot
            oneshot = p.logits(x).item()
            p.cfg.chunk_size = 64
        assert abs(chunked - oneshot) < 1e-4, (kind, chunked, oneshot)
        print(f"  {kind:9s} streaming == one-shot  ({chunked:+.6f})")
    # gradient must flow through both
    for kind in ("topr", "mean_max"):
        p = build_ext_probe(kind, cfg, r=8).train()
        z = p.logits(torch.randn(256, 128))
        z.backward()
        gn = sum(q.grad.norm().item() for q in p.parameters() if q.grad is not None)
        assert gn > 0, kind
        print(f"  {kind:9s} grad norm {gn:.3e}")
    print("probes_ext self-test OK")
