"""
Four probe families over a residual-stream sequence H of shape (T, d).

Every probe must expose `readout_direction()` -> (d,) in *residual space*, because
Phase 3 projects that vector onto SAE decoder directions. This is the contract that
keeps the project on the mechanistic side of the line rather than the AUROC side.

  LinearLastToken   s = w . H[-1] + b
                    readout = w                                   (exact)

  MeanMLP           s = MLP(mean_t H[t])
                    readout = E_x[ d s / d x ] at the pooled input (expected gradient;
                    exact for a linear model, and the standard local-linearisation of
                    a nonlinear one -- reported as such, never as "the" direction)

  EMAProbe          s = w . ema_lambda(H) + b, lambda learnable via sigmoid
                    readout = w                                   (exact)

  AttnGatedProbe    a_t = softmax_t(q . H[t] / sqrt(d))  [gate="soft"]
                        = onehot(argmax_t q . H[t])      [gate="hard"]
                    s = w . (sum_t a_t H[t]) + b
                    readout = w                                   (exact)
                    plus `query_direction()` = q, the *selection* direction, which is a
                    second, architecturally distinct object with no analogue in the
                    other families.

Note on the hard gate: an argmax over positions is the pooling rule that stops a single
informative token being averaged away by thousands of benign ones. That is the
dilution-resistance mechanism this project is testing; it is implemented here on its
own merits and is NOT attributed to any specific named method (see README).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseProbe(nn.Module):
    name: str = "base"
    needs_full_sequence: bool = True

    def pooled(self, H: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def readout_direction(self) -> torch.Tensor:
        raise NotImplementedError

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LinearLastToken(BaseProbe):
    name = "linear_last"
    needs_full_sequence = False

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def pooled(self, H: torch.Tensor) -> torch.Tensor:
        return H[..., -1, :]

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.lin(self.pooled(H)).squeeze(-1)

    @torch.no_grad()
    def readout_direction(self) -> torch.Tensor:
        return self.lin.weight.detach().flatten().clone()


class MeanMLP(BaseProbe):
    name = "mean_mlp"

    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def pooled(self, H: torch.Tensor) -> torch.Tensor:
        return H.mean(dim=-2)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.net(self.pooled(H)).squeeze(-1)

    def readout_direction(self, X: torch.Tensor | None = None) -> torch.Tensor:
        """Expected input-gradient of the score w.r.t. the pooled residual vector.

        For a linear head this recovers the weight vector exactly. For the MLP it is a
        local linearisation averaged over the data distribution -- an *effective*
        direction, and labelled that way in every downstream table.
        """
        if X is None:
            raise ValueError("MeanMLP.readout_direction requires pooled inputs X (N, d)")
        X = X.clone().requires_grad_(True)
        s = self.net(X).squeeze(-1).sum()
        g, = torch.autograd.grad(s, X)
        return g.mean(0).detach().clone()


class EMAProbe(BaseProbe):
    name = "ema"

    def __init__(self, d: int, init_lambda: float = 0.9):
        super().__init__()
        self.lin = nn.Linear(d, 1)
        # sigmoid-parameterised so lambda stays in (0, 1) without constrained optim
        inv = torch.log(torch.tensor(init_lambda / (1 - init_lambda)))
        self.lam_logit = nn.Parameter(inv.clone())

    @property
    def lam(self) -> torch.Tensor:
        return torch.sigmoid(self.lam_logit)

    def pooled(self, H: torch.Tensor) -> torch.Tensor:
        """Causal EMA over positions, normalised. Recency-weighted, unlike the mean."""
        T = H.shape[-2]
        lam = self.lam
        # weights w_t propto lam^(T-1-t): recent tokens dominate, distant ones decay
        pw = torch.arange(T - 1, -1, -1, device=H.device, dtype=H.dtype)
        w = lam ** pw
        w = w / w.sum().clamp_min(1e-9)
        return (H * w.unsqueeze(-1)).sum(dim=-2)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.lin(self.pooled(H)).squeeze(-1)

    @torch.no_grad()
    def readout_direction(self) -> torch.Tensor:
        return self.lin.weight.detach().flatten().clone()


class AttnGatedProbe(BaseProbe):
    name = "attn_gated"

    def __init__(self, d: int, gate: str = "soft", temperature: float = 1.0):
        super().__init__()
        assert gate in ("soft", "hard")
        self.q = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.lin = nn.Linear(d, 1)
        self.gate = gate
        self.temperature = temperature
        self.name = f"attn_{gate}"

    def scores(self, H: torch.Tensor) -> torch.Tensor:
        return (H @ self.q) / (H.shape[-1] ** 0.5 * self.temperature)

    def weights(self, H: torch.Tensor) -> torch.Tensor:
        s = self.scores(H)
        if self.gate == "soft":
            return F.softmax(s, dim=-1)
        # hard gate: straight-through argmax so gradients still reach q
        hard = F.one_hot(s.argmax(dim=-1), num_classes=s.shape[-1]).to(s.dtype)
        soft = F.softmax(s, dim=-1)
        return hard + soft - soft.detach()

    def pooled(self, H: torch.Tensor) -> torch.Tensor:
        return (H * self.weights(H).unsqueeze(-1)).sum(dim=-2)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.lin(self.pooled(H)).squeeze(-1)

    @torch.no_grad()
    def readout_direction(self) -> torch.Tensor:
        return self.lin.weight.detach().flatten().clone()

    @torch.no_grad()
    def query_direction(self) -> torch.Tensor:
        """The position-*selection* direction. Architecturally unique to this family."""
        return self.q.detach().clone()


def build_probe_family(d: int) -> dict[str, BaseProbe]:
    return {
        "linear_last": LinearLastToken(d),
        "mean_mlp": MeanMLP(d),
        "ema": EMAProbe(d),
        "attn_soft": AttnGatedProbe(d, gate="soft"),
        "attn_hard": AttnGatedProbe(d, gate="hard"),
    }


# --------------------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------------------
def train_probe(probe: BaseProbe, H_list: list[torch.Tensor], y: torch.Tensor,
                epochs: int = 60, lr: float = 1e-3, weight_decay: float = 1e-2,
                device: str = "cpu", seed: int = 0, verbose: bool = False) -> dict:
    """Full-batch-ish training on variable-length sequences (padded-free, list form)."""
    torch.manual_seed(seed)
    probe.to(device).train()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    y = y.to(device).float()
    n = len(H_list)
    idx = torch.arange(n)
    hist = []
    bs = 32
    for ep in range(epochs):
        perm = idx[torch.randperm(n)]
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            logits = torch.stack([probe(H_list[j].to(device)) for j in b])
            loss = F.binary_cross_entropy_with_logits(logits, y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        hist.append(tot / n)
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"      ep {ep:>3d}  loss {hist[-1]:.4f}")
    probe.eval()
    return {"final_loss": hist[-1], "loss_history": hist}


@torch.no_grad()
def score_probe(probe: BaseProbe, H_list: list[torch.Tensor], device: str = "cpu") -> torch.Tensor:
    probe.to(device).eval()
    return torch.stack([probe(H.to(device)) for H in H_list]).cpu()


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUROC with proper tie handling."""
    s = scores.detach().flatten().float()
    y = labels.detach().flatten().float()
    n_pos = int(y.sum().item())
    n_neg = int((1 - y).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    # average ranks within ties
    su = torch.unique(s)
    if len(su) < len(s):
        for v in su:
            m = s == v
            if m.sum() > 1:
                ranks[m] = ranks[m].mean()
    return ((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()
