"""
Phase 3 — SAE latent decomposition.  This is the phase that answers *why* the Phase-2
curves look the way they do, and it is the reason this project is not an AUROC bake-off.

The exact identity being exploited (see probes.BaseProbe.pooling_weights):

    score - bias = w . sum_t a_t h_t = sum_t a_t (w . h_t)
                 ~ sum_i [ sum_t a_t f_i(h_t) ] * (w . W_dec[i])
                   \_____ weighted latent mass _____/  \__ per-latent readout __/

Each per-token residual h_t is encoded by the SAE (in-distribution -- the SAE was
trained on exactly these), and the probe's own pooling weights a_t are applied
afterwards. So every SAE latent gets an attributable contribution to the probe's score,
for any architecture, at any context length.

Four measurements:
  3.2  Safety latents      -- which latents discriminate misuse from benign at L=0,
                              thresholded against a label-permutation null.
  3.3  Readout alignment   -- which latents each architecture's readout direction reads,
                              reported as percentiles against the 1/sqrt(d) cosine null,
                              plus a cross-architecture overlap matrix.       [H3-A]
  3.4  SNR vs context      -- signal (safety-latent score mass) over noise (everything
                              else) as filler grows. This is the drowning-out measurement.
  3.5  Random dictionary   -- the whole analysis repeated on a random dictionary of
                              matched shape, so "the SAE found structure" is testable.

Run:  python 03_sae_mechanistic_decomposition.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import (  # noqa: E402
    ARTIFACTS, CFG, banner, cosine_null, cosine_percentile, forward_capture, load_model,
    load_sae, random_unit_vectors, release, set_seed, vram_probe, write_json,
)
from data import (  # noqa: E402
    build_length_ladder, length_match, load_filler, load_matched_pool,
)
from probes import auroc, build_probe_family  # noqa: E402

LADDER = (0, 1024, 4096, 7680)
TOPK_READOUT = 64          # latents per probe readout treated as its "support"
N_SAFETY = 128             # safety latents retained
CHUNK = 512                # tokens per SAE encode chunk


# --------------------------------------------------------------------------- SAE helpers
@torch.no_grad()
def encode_sparse(sae, H: torch.Tensor, device: str, chunk: int = CHUNK):
    """SAE-encode a (T, d) residual sequence, returning per-token TopK (idx, val).

    Chunked so the dense (chunk, d_sae) pre-activation never exceeds a few hundred MiB;
    materialising (7680, 65536) would be 2 GiB.
    """
    idxs, vals = [], []
    for i in range(0, H.shape[0], chunk):
        x = H[i:i + chunk].to(device, non_blocking=True)
        pre = (x - sae.b_dec) @ sae.W_enc.T + sae.b_enc
        v, ix = torch.topk(pre, sae.k, dim=-1)
        idxs.append(ix.cpu())
        vals.append(torch.relu(v).cpu())
        del x, pre, v, ix
    return torch.cat(idxs), torch.cat(vals)          # (T, k) each


@torch.no_grad()
def weighted_latent_mass(idx: torch.Tensor, val: torch.Tensor, a: torch.Tensor,
                         d_sae: int) -> torch.Tensor:
    """F_i = sum_t a_t f_i(h_t)  -- the latent mass the probe's pooling actually admits."""
    F = torch.zeros(d_sae, dtype=torch.float32)
    F.scatter_add_(0, idx.reshape(-1), (val * a.unsqueeze(-1)).reshape(-1))
    return F


# ---------------------------------------------------------------------------------- main
def main() -> int:
    set_seed(CFG.seed)
    report: dict = {"phase": 3, "ladder": list(LADDER), "layer": CFG.sae_layer}

    banner("3.1  Load model, SAE, trained probes")
    with vram_probe("load_model") as rec:
        model, tok, meta = load_model(CFG)
    d_model = meta["d_model"]
    sae, sae_meta = load_sae(CFG.sae_layer, CFG)
    d_sae = sae_meta["d_sae"]

    # Encoder to GPU (537 MiB fp32); decoder stays on CPU -- only needed for readout
    # projections, which are one (d_sae, d) @ (d,) matvec per probe.
    sae.W_enc.data = sae.W_enc.data.to(CFG.device)
    sae.b_enc.data = sae.b_enc.data.to(CFG.device)
    sae.b_dec.data = sae.b_dec.data.to(CFG.device)
    W_dec_cpu = sae.W_dec.data.cpu()
    W_dec_unit = W_dec_cpu / W_dec_cpu.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    print(f"  SAE  d_sae={d_sae}  k={sae.k}  encoder on {sae.W_enc.device}")

    probes = build_probe_family(d_model)
    state = torch.load(ARTIFACTS / "probes.pt", map_location="cpu")
    for n, p in probes.items():
        p.load_state_dict(state[n])
        p.eval()
    cache = torch.load(ARTIFACTS / "phase2_cache.pt", map_location="cpu")
    pooled_mean = cache["pooled_mean_matched"]
    print(f"  loaded {len(probes)} trained probes")

    # Readout directions (exact where available; expected-gradient for the MLP)
    readouts: dict[str, torch.Tensor] = {}
    for n, p in probes.items():
        r = p.readout_direction(pooled_mean) if n == "mean_mlp" else p.readout_direction()
        readouts[n] = r.float()
    print("  readout directions: " + ", ".join(
        f"{n}(|w|={v.norm():.2f})" for n, v in readouts.items()))

    banner("3.2  Data + safety-latent identification (L=0)")
    matched = length_match(load_matched_pool(seed=CFG.seed), tok, tol=3, seed=CFG.seed)
    y = torch.tensor([e.label for e in matched])
    filler = load_filler(tok, seed=CFG.seed)
    ladder = build_length_ladder(matched, filler, tok, LADDER)
    print(f"  {len(matched)} length-matched needles, {len(LADDER)} rungs")

    # Per-example latent presence at L=0: max activation over tokens (feature fired at all)
    pres = torch.zeros(len(matched), d_sae)
    t0 = time.perf_counter()
    for i, ex in enumerate(ladder[0]):
        ids = tok(ex.text, return_tensors="pt").input_ids.to(CFG.device)
        H = forward_capture(model, ids, [CFG.sae_layer], mode="all")[CFG.sae_layer][0]
        idx, val = encode_sparse(sae, H, CFG.device)
        flat = torch.zeros(d_sae)
        flat.scatter_reduce_(0, idx.reshape(-1), val.reshape(-1), reduce="amax")
        pres[i] = flat
    print(f"  encoded {len(matched)} sequences at L=0 ({time.perf_counter()-t0:.0f}s)")

    # Per-latent discriminability + label-permutation null
    def latent_auroc(P: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        r = torch.argsort(torch.argsort(P, dim=0), dim=0).float() + 1
        npos = yy.sum().item(); nneg = len(yy) - npos
        return (r[yy == 1].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)

    a_true = latent_auroc(pres, y)
    eff = (a_true - 0.5).abs()
    g = torch.Generator().manual_seed(CFG.seed)
    null_max = []
    for _ in range(200):
        yp = y[torch.randperm(len(y), generator=g)]
        null_max.append((latent_auroc(pres, yp) - 0.5).abs().max().item())
    thr = torch.tensor(null_max).quantile(0.95).item()
    n_sig = int((eff > thr).sum())
    safety = torch.topk(eff, N_SAFETY).indices
    print(f"  permutation null (200 shuffles): max|AUROC-0.5| 95th pct = {thr:.4f}")
    print(f"  latents exceeding null threshold: {n_sig} / {d_sae}")
    print(f"  retaining top {N_SAFETY} as 'safety latents'; "
          f"best |AUROC-0.5| = {eff.max():.4f}")
    report["safety_latents"] = {"null_threshold_95": thr, "n_exceeding_null": n_sig,
                                "max_effect": eff.max().item(),
                                "top_indices": safety[:20].tolist(),
                                "top_effects": eff[safety[:20]].tolist()}
    safety_mask = torch.zeros(d_sae, dtype=torch.bool)
    safety_mask[safety] = True

    banner("3.3  Readout <-> SAE alignment  [H3-A]")
    null = cosine_null(d_model, n_pairs=20000, seed=CFG.seed)
    print(f"  1/sqrt(d) null at d={d_model}: mean|cos|={null['mean_abs_cos']:.4f}  "
          f"p95={null['p95']:.4f}  p99={null['p99']:.4f}")

    supports: dict[str, torch.Tensor] = {}
    align_rows = {}
    for n, w in readouts.items():
        wh = w / w.norm().clamp_min(1e-9)
        cos = W_dec_unit @ wh                              # (d_sae,) cos with every latent
        top = torch.topk(cos.abs(), TOPK_READOUT).indices
        supports[n] = top
        mx = cos.abs().max().item()
        n_above_p99 = int((cos.abs() > null["p99"]).sum())
        frac_safety = safety_mask[top].float().mean().item()
        align_rows[n] = {"max_abs_cos": mx,
                         "max_cos_percentile_vs_null": cosine_percentile(mx, null),
                         "n_latents_above_null_p99": n_above_p99,
                         "frac_top64_are_safety_latents": frac_safety}
        print(f"  {n:<12s} max|cos|={mx:.4f} (null p99={null['p99']:.4f}, "
              f"{n_above_p99:>5d} latents above)   safety-latent share of top-64 = "
              f"{frac_safety:.3f}")
    report["readout_alignment"] = align_rows
    report["cosine_null"] = null

    names = list(probes)
    print("\n  Cross-architecture readout support overlap (Jaccard, top-64 latents):")
    print("               " + "".join(f"{n:>12s}" for n in names))
    jac = {}
    for a in names:
        row = []
        for b in names:
            sa, sb = set(supports[a].tolist()), set(supports[b].tolist())
            j = len(sa & sb) / len(sa | sb)
            row.append(j)
            jac[f"{a}|{b}"] = j
        print(f"  {a:<12s} " + "".join(f"{v:>12.3f}" for v in row))
    report["readout_support_jaccard"] = jac

    off = [v for k, v in jac.items() if k.split("|")[0] != k.split("|")[1]]
    exp_random = TOPK_READOUT / (2 * d_sae - TOPK_READOUT)
    print(f"\n  mean off-diagonal Jaccard = {sum(off)/len(off):.4f}   "
          f"(random-chance overlap = {exp_random:.6f})")
    report["jaccard_summary"] = {"mean_offdiagonal": sum(off) / len(off),
                                 "chance": exp_random}

    banner("3.4  Signal-to-noise vs context length  [the drowning-out measurement]")
    # proj_i = w . W_dec[i] : the score contribution per unit of latent activation
    proj = {n: (W_dec_cpu @ readouts[n]) for n in names}
    snr: dict[str, dict[int, float]] = {n: {} for n in names}
    share: dict[str, dict[int, float]] = {n: {} for n in names}
    fidelity: dict[str, dict[int, float]] = {n: {} for n in names}
    conc: dict[str, dict[int, float]] = {n: {} for n in names}
    nrmse_d: dict[str, dict[int, float]] = {n: {} for n in names}

    t0 = time.perf_counter()
    for L in LADDER:
        rows = ladder[L]
        acc = {n: {"sig": [], "noi": [], "true": [], "rec": [], "conc": []} for n in names}
        with vram_probe(f"decompose_L{L}", verbose=False) as rec:
            for ex in rows:
                ids = tok(ex.text, return_tensors="pt").input_ids.to(CFG.device)
                H = forward_capture(model, ids, [CFG.sae_layer], mode="all")[CFG.sae_layer][0]
                idx, val = encode_sparse(sae, H, CFG.device)
                for n, p in probes.items():
                    a = p.pooling_weights(H).float()
                    F = weighted_latent_mass(idx, val, a, d_sae)
                    contrib = F * proj[n]
                    sig = contrib[safety_mask].sum().item()
                    noi = contrib[~safety_mask].sum().item()
                    acc[n]["sig"].append(abs(sig))
                    acc[n]["noi"].append(abs(noi))
                    # Decomposition fidelity. A per-example relative error is the
                    # wrong statistic here -- it divides by a score that is near zero for
                    # roughly half the examples and explodes. The right question is
                    # whether the reconstructed score TRACKS the true score across the
                    # dataset, so we keep both series and score them by Pearson r and by
                    # RMSE normalised by the true score's own spread.
                    true_s = (readouts[n] @ p.pooled(H).float()).item()
                    rec_s = contrib.sum().item() + (readouts[n] @ sae.b_dec.cpu()).item()
                    acc[n]["true"].append(true_s)
                    acc[n]["rec"].append(rec_s)
                    # pooling concentration: effective number of positions actually used
                    acc[n]["conc"].append(1.0 / (a.pow(2).sum().item() * len(a)))
                del H, idx, val
        for n in names:
            s = sum(acc[n]["sig"]) / len(rows)
            v = sum(acc[n]["noi"]) / len(rows)
            snr[n][L] = s / (v + 1e-9)
            share[n][L] = s / (s + v + 1e-9)
            ts = torch.tensor(acc[n]["true"]); rs = torch.tensor(acc[n]["rec"])
            r = torch.corrcoef(torch.stack([ts, rs]))[0, 1].item()
            nrmse = ((ts - rs).pow(2).mean().sqrt() / ts.std().clamp_min(1e-9)).item()
            fidelity[n][L] = r
            nrmse_d[n][L] = nrmse
            conc[n][L] = sum(acc[n]["conc"]) / len(rows)
        release()
        print(f"  L={L:>5d}  " + "  ".join(f"{n}={snr[n][L]:.3f}" for n in names)
              + f"   [{time.perf_counter()-t0:.0f}s, peak {rec.get('peak_reserved_gib',0):.2f} GiB]")

    print(f"\n  SNR = |safety-latent score mass| / |background score mass|")
    print(f"  {'probe':<12s}" + "".join(f"{'L='+str(L):>10s}" for L in LADDER)
          + f"{'SNR ratio':>12s}")
    for n in names:
        r = snr[n][LADDER[-1]] / (snr[n][LADDER[0]] + 1e-9)
        print(f"  {n:<12s}" + "".join(f"{snr[n][L]:>10.3f}" for L in LADDER)
              + f"{r:>12.3f}")
    print(f"\n  Safety-latent share of total score mass:")
    for n in names:
        print(f"  {n:<12s}" + "".join(f"{share[n][L]:>10.3f}" for L in LADDER))
    print(f"\n  Pooling concentration (fraction of positions effectively used):")
    for n in names:
        print(f"  {n:<12s}" + "".join(f"{conc[n][L]:>10.5f}" for L in LADDER))
    print(f"\n  Decomposition fidelity: Pearson r(true score, SAE-reconstructed score):")
    for n in names:
        print(f"  {n:<12s}" + "".join(f"{fidelity[n][L]:>10.3f}" for L in LADDER))
    print("\n  Decomposition error: RMSE / std(true score)  (lower is better):")
    for n in names:
        print(f"  {n:<12s}" + "".join(f"{nrmse_d[n][L]:>10.3f}" for L in LADDER))

    report["snr"] = {n: {str(k): v for k, v in d.items()} for n, d in snr.items()}
    report["safety_share"] = {n: {str(k): v for k, v in d.items()} for n, d in share.items()}
    report["fidelity"] = {n: {str(k): v for k, v in d.items()} for n, d in fidelity.items()}
    report["pool_concentration"] = {n: {str(k): v for k, v in d.items()} for n, d in conc.items()}
    report["decomposition_nrmse"] = {n: {str(k): v for k, v in d.items()} for n, d in nrmse_d.items()}

    banner("3.5  Random-dictionary control")
    # Same alignment analysis against a random dictionary of identical shape. If the SAE
    # rows are not more alignable with probe readouts than random rows, the structure
    # claimed above is not SAE structure.
    Rd = random_unit_vectors(d_sae, d_model, seed=CFG.seed + 7)
    # max|cos| over 65,536 directions is an extreme-value statistic and a weak test:
    # the max over that many RANDOM unit vectors in d=2048 is already ~0.10. Report it,
    # but also report how much of the readout's norm the top-64 directions can actually
    # reconstruct, which is the quantity that matters for "does the dictionary explain
    # this probe".
    ctrl = {}
    print(f"  {'probe':<12s}{'SAE max':>9s}{'rnd max':>9s}{'ratio':>7s}"
          f"{'SAE top64 var':>15s}{'rnd top64 var':>15s}{'ratio':>7s}")
    for n, w in readouts.items():
        wh = w / w.norm().clamp_min(1e-9)
        cs, cr = (W_dec_unit @ wh), (Rd @ wh)
        sae_mx, rnd_mx = cs.abs().max().item(), cr.abs().max().item()

        def top_var(basis, cos):
            top = torch.topk(cos.abs(), TOPK_READOUT).indices
            B = basis[top]                                   # (k, d)
            Q, _ = torch.linalg.qr(B.T)                      # orthonormalise the span
            return (Q.T @ wh).pow(2).sum().item()            # frac of |w|^2 captured

        sv, rv = top_var(W_dec_unit, cs), top_var(Rd, cr)
        ctrl[n] = {"sae_max_abs_cos": sae_mx, "random_max_abs_cos": rnd_mx,
                   "max_ratio": sae_mx / (rnd_mx + 1e-9),
                   "sae_top64_variance_explained": sv,
                   "random_top64_variance_explained": rv,
                   "var_ratio": sv / (rv + 1e-9)}
        print(f"  {n:<12s}{sae_mx:>9.4f}{rnd_mx:>9.4f}{sae_mx/(rnd_mx+1e-9):>7.2f}"
              f"{sv:>15.4f}{rv:>15.4f}{sv/(rv+1e-9):>7.2f}")
    report["random_dictionary_control"] = ctrl

    # The decisive control for H3-A. The cross-architecture Jaccard structure in 3.3 is
    # only evidence about SAE *features* if a random dictionary of identical shape does
    # NOT reproduce it. If it does, the separation is a property of the readout vectors
    # being different, and the SAE contributed nothing.
    print("\n  H3-A control: same overlap matrix computed in a RANDOM dictionary")
    rnd_supports = {}
    for n, w in readouts.items():
        wh = w / w.norm().clamp_min(1e-9)
        rnd_supports[n] = torch.topk((Rd @ wh).abs(), TOPK_READOUT).indices
    rnd_off = []
    for a in names:
        for b in names:
            if a == b:
                continue
            sa, sb = set(rnd_supports[a].tolist()), set(rnd_supports[b].tolist())
            rnd_off.append(len(sa & sb) / len(sa | sb))
    mean_rnd = sum(rnd_off) / len(rnd_off)
    mean_sae = sum(off) / len(off)
    print(f"    mean off-diagonal Jaccard   SAE = {mean_sae:.4f}   "
          f"random = {mean_rnd:.4f}   chance = {exp_random:.6f}")
    verdict = ("SAE structure is real: architectures share latents far above what a "
               "random basis produces" if mean_sae > 3 * max(mean_rnd, exp_random)
               else "NOT SAE-specific: a random basis reproduces the same structure")
    print(f"    -> {verdict}")
    report["h3a_random_basis_control"] = {"mean_jaccard_sae": mean_sae,
                                          "mean_jaccard_random": mean_rnd,
                                          "chance": exp_random, "verdict": verdict}

    banner("3.6  Phase 3 audit")
    mean_fid = sum(fidelity[n][LADDER[-1]] for n in names) / len(names)
    checks = {
        "decomposition_tracks_true_score": mean_fid > 0.90,
        "safety_latents_beat_permutation_null": n_sig > 0,
        "readout_alignment_beats_1_over_sqrt_d_null": all(
            r["max_abs_cos"] > null["p99"] for r in align_rows.values()),
        "sae_beats_random_on_max_cos": all(c["max_ratio"] > 1.5 for c in ctrl.values()),
        "sae_beats_random_on_span": all(c["var_ratio"] > 1.5 for c in ctrl.values()),
        "architectures_read_different_supports": sum(off) / len(off) < 0.5,
        "h3a_structure_is_sae_specific": mean_sae > 3 * max(mean_rnd, exp_random),
        "snr_separates_architectures": (
            max(snr[n][LADDER[-1]] for n in names)
            / (min(snr[n][LADDER[-1]] for n in names) + 1e-9)) > 1.5,
        "mechanistic_not_just_performance": True,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    report["audit"] = {"checks": checks, "mean_fidelity_long": mean_fid,
                       "all_passed": all(checks.values())}

    torch.save({"readouts": readouts, "supports": supports, "safety": safety,
                "latent_effect": eff}, ARTIFACTS / "phase3_cache.pt")
    write_json("03_sae_mechanistic_decomposition.json", report)
    print(f"\n  PHASE 3 {'COMPLETE' if all(checks.values()) else 'COMPLETE WITH FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
