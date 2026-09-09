"""Experiment 3: LSE normalisation, extreme-value drift, and cross-length calibration.

Three claims in the paper are analytic and were never measured against their own
predictions. This measures each one.

  (A) Normalised vs unnormalised LSE. Remark 3.3 says plain LogSumExp adds exactly
      H * tau * log N to the forward logit, and that because the term carries N it
      corrupts the train-to-deploy transfer specifically. Both operators are evaluated on
      identical head scores so the gap is attributable to the reduction alone, and the
      measured gap is compared against the closed form rather than merely plotted.

  (B) Extreme-value threshold drift. Remark 2.6 predicts the benign maximum grows like
      tau * sqrt(2 log N). We sample sub-Gaussian backgrounds across N in [1e2, 1e5], take
      the maximum, and fit a + b*sqrt(log N), then report b against the predicted
      tau*sqrt(2). A fit that lands elsewhere would falsify the remark.

  (C) Cross-length calibration stability. This is the one that can embarrass us. Section 4
      calibrates per length. If a single Platt map fitted at N=4096 is applied unchanged at
      1k / 32k / 128k, the drift of (B) should push the decision boundary off and inflate
      ECE and Brier. Reporting the size of that degradation is the honest way to justify
      per-length recalibration instead of asserting it.

  python experiments/exp3_drift_calibration.py [--quick]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimax_probe import ProbeConfig                                      # noqa: E402
from experiments.common import (clear, dev, make_attack_direction, make_probe,  # noqa: E402
                                save, scores_iter, synth_iter, train_probe)

D_MODEL = 2048
H = 8


# ======================================================================================
# (A) normalised vs unnormalised LogSumExp
# ======================================================================================
def lse_study(quick: bool) -> list[dict]:
    Ns = [512, 2048, 8192, 16_384] + ([] if quick else [65_536, 131_072])
    taus = [0.5, 1.0]
    torch.manual_seed(11)
    rows = []
    for N in Ns:
        for tau in taus:
            # One fixed score tensor feeds BOTH operators, so the only difference between
            # the two numbers below is the reduction, not the sample.
            s = torch.randn(H, N, device=dev())
            lse_unnorm = tau * torch.logsumexp(s / tau, dim=1)          # (H,)
            lse_norm = tau * (torch.logsumexp(s / tau, dim=1) - math.log(N))
            hard = s.amax(dim=1)
            gap = float((lse_unnorm - lse_norm).sum())                  # summed over heads
            rows.append({
                "N": N, "tau": tau, "H": H,
                "logit_unnormalised": float(lse_unnorm.sum()),
                "logit_normalised": float(lse_norm.sum()),
                "logit_hard_max": float(hard.sum()),
                "measured_gap": gap,
                "predicted_gap_H_tau_logN": H * tau * math.log(N),
                "abs_error": abs(gap - H * tau * math.log(N)),
            })
            print(f"  N={N:>7} tau={tau:<4} gap={gap:8.3f}  "
                  f"predicted={H * tau * math.log(N):8.3f}")
            del s
            clear()
    return rows


# ======================================================================================
# (B) extreme-value drift of the benign maximum
# ======================================================================================
def drift_study(quick: bool) -> dict:
    tau = 1.0
    Ns = [100, 316, 1000, 3162, 10_000, 31_623, 100_000]
    trials = 64 if quick else 256
    torch.manual_seed(23)
    rows = []
    for N in Ns:
        mx = torch.stack([torch.randn(N, device=dev()).max() for _ in range(trials)])
        rows.append({"N": N, "mean_max": float(mx.mean()), "std_max": float(mx.std()),
                     "sqrt_2logN": math.sqrt(2 * math.log(N))})
        print(f"  N={N:>7} E[max]={rows[-1]['mean_max']:.4f}  "
              f"sqrt(2 log N)={rows[-1]['sqrt_2logN']:.4f}")
        del mx
        clear()

    # least squares fit of  E[max] = a + b * sqrt(log N)
    x = torch.tensor([math.sqrt(math.log(r["N"])) for r in rows])
    y = torch.tensor([r["mean_max"] for r in rows])
    A = torch.stack([torch.ones_like(x), x], dim=1)
    coef = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.flatten()
    a, b = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"points": rows, "fit_intercept": a, "fit_slope": b,
            "predicted_slope_tau_sqrt2": tau * math.sqrt(2),
            "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")}


# ======================================================================================
# (C) cross-length calibration with a single held-out Platt map
# ======================================================================================
def platt_fit(z: torch.Tensor, y: torch.Tensor, steps: int = 300) -> tuple[float, float]:
    a = torch.ones(1, device=z.device, requires_grad=True)
    b = torch.zeros(1, device=z.device, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.05, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * z + b, y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(a.detach()), float(b.detach())


def ece(p: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins on the probability axis."""
    e, n = 0.0, p.numel()
    edges = torch.linspace(0, 1, bins + 1, device=p.device)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if int(m.sum()) == 0:
            continue
        e += float(m.float().mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return e


def calibration_study(quick: bool) -> dict:
    # Weak signal on purpose. At S=0.30 the probe separates the classes perfectly and
    # every Brier score is 0.0000 at every length, so the study cannot show a transfer
    # penalty whether or not one exists. S=0.15 leaves the probe imperfect, which is the
    # regime in which calibration is worth arguing about.
    STRENGTH = 0.15
    cfg = ProbeConfig(d_model=D_MODEL, hidden=512, n_heads=H, chunk_size=4096)
    atk = make_attack_direction(D_MODEL)
    probe = train_probe("multimax", cfg, atk, N_train=512, k_train=8, strength=STRENGTH,
                        n_per_class=48 if quick else 96, epochs=8 if quick else 12,
                        seed=5)

    def logits_at(N: int, n: int, seed: int):
        # Streamed: materialising 2n sequences at N=32,768 needs ~32 GiB and OOMs on this
        # 8 GiB card, which is exactly how the first run of this study died.
        zp = scores_iter(probe, synth_iter(n, N, 8, atk, True, STRENGTH, seed))
        zn = scores_iter(probe, synth_iter(n, N, 0, atk, False, STRENGTH, seed + 1))
        z = torch.cat([zp, zn])
        y = torch.cat([torch.ones_like(zp), torch.zeros_like(zn)])
        clear()
        return z, y

    n_cal = 32 if quick else 64
    z_fit, y_fit = logits_at(4096, n_cal, 101)
    a, b = platt_fit(z_fit, y_fit)
    print(f"  Platt fitted at N=4096: a={a:.4f} b={b:.4f}")

    Ns = [1024, 32_768] + ([] if quick else [131_072])
    rows = []
    for N in [4096] + Ns:
        z, y = logits_at(N, n_cal, 200 + N % 1000)
        p_raw = torch.sigmoid(z)
        p_tr = torch.sigmoid(a * z + b)                       # transferred map
        a_l, b_l = platt_fit(z, y)                            # per-length oracle
        p_lo = torch.sigmoid(a_l * z + b_l)
        rows.append({
            "N": N, "is_fit_length": N == 4096,
            "mean_logit": float(z.mean()),
            "brier_uncalibrated": float(((p_raw - y) ** 2).mean()),
            "brier_transferred": float(((p_tr - y) ** 2).mean()),
            "brier_per_length": float(((p_lo - y) ** 2).mean()),
            "ece_uncalibrated": ece(p_raw, y), "ece_transferred": ece(p_tr, y),
            "ece_per_length": ece(p_lo, y),
            "acc_transferred": float(((p_tr > 0.5).float() == y).float().mean()),
            "acc_per_length": float(((p_lo > 0.5).float() == y).float().mean()),
        })
        r = rows[-1]
        print(f"  N={N:>7} mean logit={r['mean_logit']:7.2f}  "
              f"Brier transfer={r['brier_transferred']:.4f} "
              f"per-length={r['brier_per_length']:.4f}  "
              f"ECE transfer={r['ece_transferred']:.4f}")
        del z, y
        clear()
    return {"platt_at_4096": {"a": a, "b": b}, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("== (A) normalised vs unnormalised LogSumExp ==")
    lse = lse_study(args.quick)
    print("\n== (B) extreme-value drift of the benign maximum ==")
    drift = drift_study(args.quick)
    print(f"  fit: E[max] = {drift['fit_intercept']:.4f} + "
          f"{drift['fit_slope']:.4f} sqrt(log N)   R^2={drift['r_squared']:.4f}")
    print(f"  predicted slope tau*sqrt(2) = {drift['predicted_slope_tau_sqrt2']:.4f}")
    print("\n== (C) cross-length calibration transfer ==")
    cal = calibration_study(args.quick)

    worst = max((r for r in cal["rows"] if not r["is_fit_length"]),
                key=lambda r: r["brier_transferred"])
    verdicts = {
        "lse_max_abs_error_vs_closed_form": max(r["abs_error"] for r in lse),
        "lse_gap_matches_H_tau_logN": all(r["abs_error"] < 1e-3 for r in lse),
        "drift_slope_measured": drift["fit_slope"],
        "drift_slope_predicted": drift["predicted_slope_tau_sqrt2"],
        "drift_r_squared": drift["r_squared"],
        "worst_transfer_N": worst["N"],
        "worst_transfer_brier": worst["brier_transferred"],
        "worst_per_length_brier": worst["brier_per_length"],
        # A RATIO test is worthless here. Both Brier scores can be ~1e-8, in which case a
        # 1000x ratio is numerical noise and reporting "recalibration needed" from it
        # would be a false positive. Require the gap to be large enough to matter to a
        # deployment (0.01 Brier) before claiming anything.
        "per_length_recalibration_needed":
            (worst["brier_transferred"] - worst["brier_per_length"]) > 0.01,
        "logit_drift_1k_to_max": (
            max(r["mean_logit"] for r in cal["rows"])
            - min(r["mean_logit"] for r in cal["rows"])),
        "transfer_penalty_detectable": (
            worst["brier_transferred"] - worst["brier_per_length"]) > 0.01,
    }
    print("\nverdicts:", verdicts)
    save("exp3_drift_calibration.json",
         {"lse": lse, "drift": drift, "calibration": cal, "verdicts": verdicts,
          "config": {"d_model": D_MODEL, "H": H, "quick": args.quick}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
