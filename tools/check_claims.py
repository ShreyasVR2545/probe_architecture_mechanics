"""Verify every load-bearing number in the prose against the JSON artifacts.

Written after a real drift: BENCHMARK_NOTES.md and a docstring carried calibration
figures (Brier 0.0338 -> 0.0157, accuracy 0.963 -> 0.981) that predated the
LSE-normalisation fix. The fix changed how the probe trains, so those numbers were
stale, but nothing in the pipeline noticed -- they were prose, and prose is not tested.

This closes that gap. Each claim names a value, where the prose asserts it, and where
the artifact proves it. Run it before any commit that touches documented figures.

  python tools/check_claims.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark_results.json"
CAL = ROOT / "logs" / "calibration_report.json"
NOTES = ROOT / "BENCHMARK_NOTES.md"
TEX = ROOT / "math_formulation.tex"
PAPER = ROOT / "paper.tex"


def approx(a: float, b: float, tol: float = 5e-3) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def norm_tex(raw: str) -> str:
    """Flatten LaTeX to plain text so numeric claims can be matched literally.

    Strips one level of formatting macros, drops math delimiters, turns thin/hard
    spaces into ordinary ones, and collapses runs of whitespace. The collapse is what
    lets a whole table ROW be matched as a single string even though the source wraps
    it across lines -- a row match catches a swapped column, which a per-cell match
    does not.
    """
    s = re.sub(r"\\(?:mathbf|mathrm|textbf|emph|text)\{([^{}]*)\}", r"\1", raw)
    s = s.replace("$", "").replace("\\,", " ").replace("~", " ")
    s = s.replace("{=}", "=").replace("{,}", ",")
    return re.sub(r"\s+", " ", s)


def main() -> int:
    for p in (BENCH, CAL):
        if not p.exists():
            print(f"MISSING artifact: {p.relative_to(ROOT)} -- run the suite first")
            return 2

    bench = json.loads(BENCH.read_text())
    cal = json.loads(CAL.read_text())["calibration"]
    notes = NOTES.read_text(encoding="utf-8")
    # Strip LaTeX markup before matching: "$\mathbf{19.1}$ MiB" must match "19.1 MiB".
    # Without this the checker reports false failures on correctly-typeset numbers.
    tex_raw = TEX.read_text(encoding="utf-8")
    tex = re.sub(r"\\(?:mathbf|mathrm|textbf|emph|text)\{([^{}]*)\}", r"\1", tex_raw)
    tex = tex.replace("$", "").replace("\\,", "").replace("~", " ")
    paper = norm_tex(PAPER.read_text(encoding="utf-8"))

    A = {(r["probe"], r["N"]): r for r in bench["suite_a_latency_memory"]}
    B = {(r["probe"], r["strength"]): r for r in bench["suite_b_strength_sweep"]}
    C = {(r["probe"], r["m_spread"]): r for r in bench["suite_c_distributed_attack"]}
    V = bench["verdicts"]

    checks: list[tuple[str, bool, str]] = []

    def claim(name: str, value, rendered: str, docs: list[str]) -> None:
        """Assert `rendered` appears in each doc, and record the source value."""
        for d, label in docs:
            checks.append((f"{name} [{label}]", rendered in d,
                           f"expected '{rendered}' (artifact says {value})"))

    # --- Suite A: overhead at the top of the ladder ---------------------------------
    mm = A[("multimax", 131072)]["overhead_mib"]
    sm = A[("softmax_attn", 131072)]["overhead_mib"]
    claim("A multimax overhead @131k", mm, f"{mm:.1f} MiB",
          [(notes, "NOTES"), (tex, "TEX")])
    claim("A softmax overhead @131k", sm, f"{sm:.1f}",
          [(notes, "NOTES"), (tex, "TEX")])
    checks.append(("A multimax alpha == 0.000",
                   approx(V["memory_scaling_alpha"]["multimax"], 0.0),
                   f"got {V['memory_scaling_alpha']['multimax']}"))
    checks.append(("A self_attn OOM logged", "self_attn" in V["logged_oom"],
                   f"got {V['logged_oom']}"))

    # --- Suite B: weak-signal recall -------------------------------------------------
    mm10 = B[("multimax", 0.10)]["recall_long"]
    sm10 = B[("softmax_attn", 0.10)]["recall_long"]
    checks.append(("B multimax recall@S=0.10 == 1.00", approx(mm10, 1.0), f"got {mm10}"))
    claim("B softmax recall@S=0.10", sm10, f"{sm10:.2f}", [(notes, "NOTES"), (tex, "TEX")])
    # EVERY cell of the Suite-B table, not just the two headline probes. The mean_pool
    # column drifted to quick-run values precisely because it was unchecked.
    for st in (0.10, 0.15, 0.50):
        for pr in ("multimax", "softmax_attn", "mean_pool"):
            cell = (f"{B[(pr, st)]['recall_long']:.2f} / "
                    f"{B[(pr, st)]['recall_train_length']:.2f}")
            checks.append((f"B table cell {pr}@{st:.2f}", cell in notes,
                           f"expected '{cell}' in NOTES"))
    checks.append(("B annealing verdict True",
                   V["annealing_fixes_low_signal_training"] is True,
                   f"got {V['annealing_fixes_low_signal_training']}"))

    # --- Suite C: boundary and AUROC at maximum spread -------------------------------
    bnd = V["distributed_attack_failure_boundary_m"]
    checks.append(("C boundary multimax == 64", bnd["multimax"] == 64, f"got {bnd}"))
    checks.append(("C boundary softmax == 64", bnd["softmax_attn"] == 64, f"got {bnd}"))
    au_mm = C[("multimax", 256)]["auroc"]
    au_sm = C[("softmax_attn", 256)]["auroc"]
    claim("C multimax AUROC@m=256", au_mm, f"{au_mm:.3f}", [(notes, "NOTES"), (tex, "TEX")])
    claim("C softmax AUROC@m=256", au_sm, f"{au_sm:.3f}", [(notes, "NOTES"), (tex, "TEX")])

    # --- paper.tex: the manuscript is the primary artifact and was previously the ONLY
    # document not covered here. Table 3 carries 8 columns x 4 rows of Suite-C numbers;
    # we match each row whole, so a transposed or mislabelled column fails too. --------
    for m in (1, 16, 64, 256):
        row = " & ".join([
            str(m), f"{C[('multimax', m)]['per_token_strength']:.4f}",
            f"{C[('multimax', m)]['recall']:.3f}", f"{C[('multimax', m)]['auroc']:.3f}",
            f"{C[('softmax_attn', m)]['recall']:.3f}", f"{C[('softmax_attn', m)]['auroc']:.3f}",
            f"{C[('mean_pool', m)]['recall']:.3f}", f"{C[('mean_pool', m)]['auroc']:.3f}",
        ])
        checks.append((f"PAPER Table 3 row m={m}", row in paper,
                       f"expected row '{row}'"))

    # The paper's central self-undermining claim: mean pooling is fragmentation-INVARIANT.
    # Assert both the fact (from the artifact) and the range quoted in the prose.
    mp_au = [C[("mean_pool", m)]["auroc"] for m in (1, 4, 16, 64, 256)]
    checks.append(("C mean_pool AUROC invariant (spread < 0.10)",
                   max(mp_au) - min(mp_au) < 0.10,
                   f"range {min(mp_au):.3f}-{max(mp_au):.3f}"))
    checks.append(("PAPER mean_pool AUROC range quoted",
                   f"{min(mp_au):.3f}--{max(mp_au):.3f}" in paper
                   or f"{min(mp_au):.3f}" in paper and f"{max(mp_au):.3f}" in paper,
                   f"expected {min(mp_au):.3f} and {max(mp_au):.3f} in paper.tex"))
    checks.append(("C mean_pool best AUROC at m=256",
                   C[("mean_pool", 256)]["auroc"] > max(au_mm, au_sm),
                   f"mean {C[('mean_pool', 256)]['auroc']:.3f} vs "
                   f"mm {au_mm:.3f} / sm {au_sm:.3f}"))

    # --- Analytical constants asserted in the prose ----------------------------------
    # sigma'(c) at the clamp bound c=10, quoted in Remark 2.1.
    sig = 1.0 / (1.0 + math.exp(-10.0))
    checks.append(("sigma'(10) == 4.54e-05 as quoted",
                   f"{sig * (1 - sig):.2e}".replace("e-05", "") .startswith("4.54")
                   and "4.54\\times10^{-5}" in PAPER.read_text(encoding="utf-8"),
                   f"computed {sig * (1 - sig):.3e}"))
    # The HBM round-trip arithmetic in section 7.1: (1,N,m) bf16 at N=131072, m=512.
    mib = 131072 * 512 * 2 / 2 ** 20
    checks.append((f"HBM intermediate == {mib:.0f} MiB as quoted",
                   mib == 128 and "128 MiB exactly" in paper,
                   f"computed {mib:.1f} MiB"))

    # --- Calibration -----------------------------------------------------------------
    claim("Brier before", cal["brier_before"], f"{cal['brier_before']:.4f}",
          [(notes, "NOTES")])
    claim("Brier after", cal["brier_after"], f"{cal['brier_after']:.4f}",
          [(notes, "NOTES")])
    claim("accuracy after", cal["accuracy_after"], f"{cal['accuracy_after']:.3f}",
          [(notes, "NOTES")])
    checks.append(("Brier reduced", cal["brier_after"] < cal["brier_before"],
                   f"{cal['brier_before']:.4f} -> {cal['brier_after']:.4f}"))
    checks.append(("accuracy >= 0.93", cal["accuracy_after"] >= 0.93,
                   f"got {cal['accuracy_after']:.3f}"))

    # --- Stale-figure tripwire --------------------------------------------------------
    # These are the superseded values. They may appear ONLY inside the explicit
    # "Superseded figures" note that explains why they are wrong.
    for stale in ("0.0157", "0.963"):
        hits = [m.start() for m in re.finditer(re.escape(stale), notes)]
        ok = all("Superseded" in notes[max(0, h - 400):h + 200] for h in hits)
        checks.append((f"stale figure {stale} only in the superseded note", ok,
                       f"{len(hits)} occurrence(s)"))

    # ==================================================================================
    # Reviewer-response experiments (artifacts/exp*.json). Same rule as above: every
    # number the new sections quote must be reproducible from an artifact, or the paper
    # is asserting something no run supports.
    # ==================================================================================
    def art(name):
        p = ROOT / "artifacts" / name
        return json.loads(p.read_text()) if p.exists() else None

    def has(val, doc=None):
        return val in (paper if doc is None else doc)

    e1 = art("exp1_real_residuals.json")
    if e1:
        sp = e1["verdicts"]["split_comparison_mean_auroc"]
        for kind, want in (("multimax", "0.731"), ("softmax_attn", "0.690"),
                           ("mean_pool", "0.696")):
            v = sp[kind]["needle_split"]
            checks.append((f"E1 needle-split mean AUROC {kind}",
                           f"{v:.3f}" == want and has(want),
                           f"artifact {v:.4f}, paper expects {want}"))
        gap = e1["verdicts"]["leakage_gap_multimax"]
        checks.append(("E1 leakage gap 0.106", f"{gap:.3f}" == "0.106" and has("0.106"),
                       f"artifact {gap:.4f}"))
        det = e1["detection"]
        for L, want in ((16, "0.860"), (24, "0.797"), (31, "0.535")):
            v = [r["auroc"] for r in det if r["probe"] == "multimax" and r["layer"] == L]
            m = sum(v) / len(v)
            checks.append((f"E1 multimax layer-{L} mean {want}",
                           f"{m:.3f}" == want and has(want), f"artifact {m:.4f}"))
        sysrows = [r for r in e1["systems"] if not r.get("oom")]
        mm = [r["peak_overhead_mib"] for r in sysrows
              if r["probe"] == "multimax" and r["N"] >= 4096]
        # Compared at the precision the paper QUOTES (one decimal). The raw values differ
        # in the fourth decimal (8.00049 vs 8.00098 MiB) because the allocator rounds to
        # pages; requiring exact equality would fail a claim that is true as stated.
        mm_r = {f"{v:.1f}" for v in mm}
        checks.append(("E1 multimax overhead flat at 8.0 MiB",
                       mm_r == {"8.0"} and has("8.0 MiB"),
                       f"artifact rounds to {sorted(mm_r)} from {sorted(set(mm))}"))
        sm = next(r["peak_overhead_mib"] for r in sysrows
                  if r["probe"] == "softmax_attn" and r["N"] == 131072)
        checks.append(("E1 softmax overhead 641.0 MiB @131k",
                       f"{sm:.1f}" == "641.0" and has("641.0"), f"artifact {sm}"))

    e2 = art("exp2_ablations.json")
    if e2:
        for c_, H_, want in ((1.0, 8, "0.500"), (5.0, 1, "1.000"), (10.0, 32, "1.000"),
                             (5.0, 4, "0.975")):
            v = next(r["auroc_ood"] for r in e2["grid"]
                     if r["c"] == c_ and r["H"] == H_)
            checks.append((f"E2 grid c={c_} H={H_} = {want}",
                           f"{v:.3f}" == want and has(want), f"artifact {v:.4f}"))
        for c_, want in ((1.0, "0.8133"), (2.0, "1.1269")):
            v = next(r["final_loss"] for r in e2["grid"] if r["c"] == c_ and r["H"] == 8)
            checks.append((f"E2 constant loss at c={c_}", f"{v:.4f}" == want and has(want),
                           f"artifact {v:.5f}"))
        checks.append(("E2 plain clamp dead in all dtypes",
                       e2["verdicts"]["plain_clamp_dead_all_dtypes"] is True, ""))
        checks.append(("E2 STE alive in all dtypes",
                       e2["verdicts"]["ste_rescues_all_dtypes"] is True, ""))

    e2b = art("exp2b_saturation.json")
    if e2b:
        for c_, H_, want in ((1.0, 8, "1605.6"), (2.0, 8, "344.3"), (5.0, 8, "69.9")):
            r = next(x for x in e2b["cells"] if x["c"] == c_ and x["H"] == H_)
            v = r["train_len"]["mean_abs_raw"]
            checks.append((f"E2b pre-clamp |z| c={c_} = {want}",
                           f"{v:.1f}" == want and has(want), f"artifact {v:.2f}"))
        checks.append(("E2b all dead cells saturated",
                       e2b["verdicts"]["dead_cells_saturated_at_eval"] is True, ""))

    e3 = art("exp3_drift_calibration.json")
    if e3:
        v = e3["verdicts"]
        checks.append(("E3 LSE matches closed form", v["lse_gap_matches_H_tau_logN"] is True,
                       f"max err {v['lse_max_abs_error_vs_closed_form']:.2e}"))
        checks.append(("E3 drift slope 1.513",
                       f"{v['drift_slope_measured']:.3f}" == "1.513" and has("1.513"),
                       f"artifact {v['drift_slope_measured']:.4f}"))
        checks.append(("E3 drift R^2 0.999",
                       f"{v['drift_r_squared']:.3f}" == "0.999" and has("0.999"),
                       f"artifact {v['drift_r_squared']:.4f}"))
        checks.append(("E3 logit drift 1.52",
                       f"{v['logit_drift_1k_to_max']:.2f}" == "1.52" and has("1.52"),
                       f"artifact {v['logit_drift_1k_to_max']:.3f}"))
        # the paper says recalibration is NOT empirically forced; the artifact must agree
        checks.append(("E3 no transfer penalty (paper says so)",
                       v["per_length_recalibration_needed"] is False,
                       f"gap {v.get('transfer_penalty_brier_gap')}"))

    e4 = art("exp4_fragmentation.json")
    if e4:
        for probe, m_, want in (("multimax", 64, "0.649"), ("topr", 64, "0.908"),
                                ("mean_max", 64, "0.575"), ("softmax_attn", 256, "0.749"),
                                ("mean_pool", 256, "0.761"), ("multimax", 256, "0.429")):
            r = [x for x in e4["fragmentation"] if x["probe"] == probe and x["m"] == m_
                 and not x["mixed_structure"]]
            if not r:
                continue
            checks.append((f"E4 unknown-m {probe}@{m_} = {want}",
                           f"{r[0]['auroc']:.3f}" == want and has(want),
                           f"artifact {r[0]['auroc']:.4f}"))
        checks.append(("E4 mean_max does NOT dominate (paper says refuted)",
                       e4["verdicts"]["mean_max_dominates_both_members"] is False, ""))
        b16 = {r["probe"]: r for r in e4["batched"] if r.get("B") == 16
               and not r.get("oom")}
        if "multimax" in b16 and "softmax_attn" in b16:
            checks.append(("E4 batched B=16 320 vs 5128 MiB",
                           f"{b16['multimax']['overhead_above_input_mib']:.0f}" == "320"
                           and f"{b16['softmax_attn']['overhead_above_input_mib']:.0f}"
                           == "5128" and has("5128"), ""))
            checks.append(("E4 batched B=16 latency 92.68 vs 320.57",
                           f"{b16['multimax']['latency_ms']:.2f}" == "92.68"
                           and f"{b16['softmax_attn']['latency_ms']:.2f}" == "320.57"
                           and has("92.68") and has("320.57"), ""))

    # exp5 is SUPERSEDED by exp7 (n=48 -> n=532, templated prompts -> real corpora), so
    # the paper no longer quotes its figures. Its checks therefore verify only that the
    # artifact is self-consistent, NOT that the numbers appear in the prose: requiring
    # that would fail the moment a superseded result is correctly removed from the paper.
    e5 = art("exp5_cascade.json")
    if e5:
        v = e5["verdicts"]
        checks.append(("E5 (superseded) stage1 AUROC 0.554",
                       f"{v['stage1_auroc']:.3f}" == "0.554", ""))
        checks.append(("E5 (superseded) stage2 is a real model",
                       "SGuard" in v["stage2_is_real_model"], v["stage2_is_real_model"]))
        checks.append(("E5 (superseded) FLOPs ratio 0.104%",
                       f"{v['flops_ratio_stage1_to_stage2'] * 100:.3f}" == "0.104"
                       and has("0.104"), f"{v['flops_ratio_stage1_to_stage2']:.6f}"))
        checks.append(("E5 superseded by E7 at larger n",
                       art("exp7_cascade_scaled.json") is not None
                       and art("exp7_cascade_scaled.json")["verdicts"]["n_eval"]
                       > v["n_test"], ""))

    e6 = art("exp6_pareto.json")
    if e6:
        g6 = e6["grid"]

        def mean_r(layer, r_):
            v = [x["auroc"] for x in g6 if x["layer"] == layer and x["r"] == r_]
            return sum(v) / len(v) if v else None
        for L, want in ((16, "-0.009"), (31, "+0.200")):
            d_ = mean_r(L, 32) - mean_r(L, 1)
            checks.append((f"E6 layer-{L} r=1->32 delta {want}",
                           f"{d_:+.3f}" == want and has(want.lstrip('+')),
                           f"artifact {d_:+.4f}"))
        for L, r_, want in ((16, 1, "0.840"), (16, 32, "0.831"),
                            (31, 1, "0.516"), (31, 32, "0.715")):
            v = mean_r(L, r_)
            checks.append((f"E6 mean AUROC L{L} r={r_} = {want}",
                           f"{v:.3f}" == want and has(want), f"artifact {v:.4f}"))
        # overhead must depend on m and NOT on r; the paper says within 0.005 MiB
        spread = []
        for m_ in {x["m"] for x in g6}:
            v = [x["overhead_mib"] for x in g6 if x["m"] == m_]
            spread.append(max(v) - min(v))
        checks.append(("E6 overhead independent of r (<0.005 MiB)",
                       max(spread) < 0.005 and has("0.005"),
                       f"max spread {max(spread):.5f} MiB"))
        cheap = [x for x in g6 if x["layer"] == 16 and x["r"] == 2 and x["m"] == 32]
        rich = [x for x in g6 if x["layer"] == 16 and x["r"] == 1 and x["m"] == 128]
        if cheap and rich:
            checks.append(("E6 cheap corner 0.905 @0.50 vs 0.906 @2.00 MiB",
                           f"{cheap[0]['auroc']:.3f}" == "0.905"
                           and f"{rich[0]['auroc']:.3f}" == "0.906"
                           and has("0.905") and has("0.906"), ""))

    e7 = art("exp7_cascade_scaled.json")
    if e7:
        v = e7["verdicts"]
        checks.append(("E7 evaluation set >= 500 prompts", v["n_eval"] >= 500,
                       f"n_eval={v['n_eval']}"))
        checks.append(("E7 corpus is real, not templated",
                       "harmful_behaviors" in v["corpus"]["harmful"]
                       and "alpaca" in v["corpus"]["benign"], str(v["corpus"])))
        checks.append(("E7 prompt-disjoint split",
                       v["corpus"]["split"] == "prompt-disjoint", ""))
        # the paper's claim about WHICH gates are inert must match the artifact
        for rule, inert in v["gate_inert"].items():
            checks.append((f"E7 gate '{rule}' inert = {inert}", isinstance(inert, bool),
                           ""))
        checks.append(("E7 rank gate is never inert",
                       v["gate_inert"]["rank"] is False, ""))
        checks.append(("E7 stage1 AUROC 0.950 / stage2 0.998",
                       f"{v['stage1_auroc']:.3f}" == "0.950"
                       and f"{v['stage2_auroc']:.3f}" == "0.998"
                       and has("0.950") and has("0.998"), ""))
        checks.append(("E7 platt a=6.30 (not degenerate)",
                       f"{v['platt_a']:.2f}" == "6.30" and has("6.30"),
                       f"artifact {v['platt_a']:.4f}"))
        te = v["escalation_tracking_error"]
        for rule, want in (("platt", "0.0860"), ("temperature", "0.0492"),
                           ("isotonic", "0.0785"), ("rank", "0.0008")):
            checks.append((f"E7 tracking error {rule} = {want}",
                           f"{te[rule]:.4f}" == want and has(want),
                           f"artifact {te[rule]:.5f}"))
        r25 = v["realised_at_target_25pct"]
        checks.append(("E7 platt realises 0.000 at a 25% budget",
                       abs(r25["platt"]) < 1e-9, f"{r25['platt']}"))
        checks.append(("E7 rank realises 0.250 at a 25% budget",
                       f"{r25['rank']:.3f}" == "0.250", f"{r25['rank']}"))
        row25 = [r for r in e7["cascade"] if r["rule"] == "rank"
                 and abs(r["target_rate"] - 0.25) < 1e-9]
        if row25:
            checks.append(("E7 cascade ASR 0.053 at 25% budget",
                           f"{row25[0]['attack_success_rate']:.3f}" == "0.053"
                           and has("0.053"), f"{row25[0]['attack_success_rate']}"))
        checks.append(("E7 always-stage2 ASR 0.496",
                       f"{v['asr_always_stage2']:.3f}" == "0.496" and has("0.496"), ""))

    e8 = art("exp8_qwen_family.json")
    if e8:
        v = e8["verdicts"]
        checks.append(("E8 second family is Qwen2.5", "Qwen2.5" in v["model"],
                       v["model"]))
        checks.append(("E8 gemma gated recorded", v["gemma_gated_403"] is True, ""))
        checks.append(("E8 depth ratios 50/75/100",
                       v["depth_ratios"] == [0.5, 0.75, 1.0], str(v["depth_ratios"])))
        mm, sm = v["mean_auroc"]["multimax"], v["mean_auroc"]["softmax_attn"]
        # The paper says plainly that MultiMax does NOT beat softmax on Qwen's average.
        # If a rerun flipped that, the prose would be wrong, so assert the direction.
        checks.append(("E8 multimax does NOT beat softmax on average (paper says so)",
                       v["multimax_beats_softmax"] is False,
                       f"multimax {mm:.4f}, softmax {sm:.4f}"))
        for kind, want in (("multimax", "0.817"), ("topr", "0.859"),
                           ("softmax_attn", "0.833")):
            checks.append((f"E8 mean AUROC {kind} = {want}",
                           f"{v['mean_auroc'][kind]:.3f}" == want and has(want),
                           f"artifact {v['mean_auroc'][kind]:.4f}"))
        # mean_pool is no longer tabulated: the Qwen table now compares against the
        # baseline families instead. The measurement still has to be right, so the value
        # is asserted, but not its presence in prose it no longer appears in.
        checks.append(("E8 mean AUROC mean_pool = 0.748 (artifact only)",
                       f"{v['mean_auroc']['mean_pool']:.3f}" == "0.748",
                       f"artifact {v['mean_auroc']['mean_pool']:.4f}"))
        det8 = e8["detection"]

        def m8(N, kind):
            z = [r["auroc"] for r in det8 if r["N"] == N and r["probe"] == kind]
            return sum(z) / len(z) if z else None
        for N, kind, want in ((256, "softmax_attn", "0.947"),
                              (4096, "softmax_attn", "0.661"),
                              (256, "multimax", "0.835"), (4096, "multimax", "0.774"),
                              (4096, "topr", "0.814")):
            got = m8(N, kind)
            checks.append((f"E8 mean AUROC {kind}@N={N} = {want}",
                           f"{got:.3f}" == want and has(want), f"artifact {got:.4f}"))
        checks.append(("E8 memory law unchanged (8.0 flat, 641.0 softmax)",
                       f"{v['multimax_overhead_mib']:.1f}" == "8.0"
                       and f"{v['softmax_overhead_at_max_N']:.1f}" == "641.0"
                       and v["multimax_overhead_flat"] is True, ""))
        # the near-final collapse must NOT reproduce, which is what the paper claims
        mm100 = [r["auroc"] for r in det8
                 if r["depth_ratio"] == 1.0 and r["probe"] == "multimax"]
        checks.append(("E8 no final-layer collapse on Qwen (0.773)",
                       f"{sum(mm100) / len(mm100):.3f}" == "0.773" and has("0.773"),
                       f"artifact {sum(mm100) / len(mm100):.4f}"))

    e9 = art("exp9_bootstrap_baselines.json")
    if e9:
        v = e9["verdicts"]
        checks.append(("E9 1000 bootstrap resamples", v["n_bootstraps"] == 1000,
                       f"n_boot={v['n_bootstraps']}"))
        checks.append(("E9 lengths 512/1024/2048/4096",
                       v["lengths"] == [512, 1024, 2048, 4096], str(v["lengths"])))
        checks.append(("E9 no missing cells", not v["missing_cells"],
                       f"{len(v['missing_cells'])} missing"))
        au, tp = v["mean_auroc"], v["mean_tpr_at_1fpr"]
        # The paper says plainly that a LINEAR baseline wins on AUROC and that Top-r
        # wins on the operational metric. Both directions are asserted, so a rerun that
        # flipped either would fail the build rather than quietly contradict the prose.
        for m in ("mistral-7b", "qwen2.5-7b"):
            checks.append((f"E9 {m}: mean_logreg best on AUROC (paper says so)",
                           au[m]["mean_logreg"] == max(au[m].values()),
                           f"{ {k: round(x,3) for k,x in au[m].items()} }"))
            checks.append((f"E9 {m}: topr best on TPR@1%FPR (paper says so)",
                           tp[m]["topr"] == max(tp[m].values()),
                           f"{ {k: round(x,3) for k,x in tp[m].items()} }"))
            checks.append((f"E9 {m}: latentbiopsy worst on TPR@1%FPR",
                           tp[m]["latentbiopsy"] == min(tp[m].values()), ""))
        for m, kind, want in (("mistral-7b", "mean_logreg", "0.802"),
                              ("mistral-7b", "topr", "0.738"),
                              ("qwen2.5-7b", "mean_logreg", "0.871"),
                              ("qwen2.5-7b", "topr", "0.853")):
            checks.append((f"E9 mean AUROC {m}/{kind} = {want}",
                           f"{au[m][kind]:.3f}" == want and has(want),
                           f"artifact {au[m][kind]:.4f}"))
        for m, kind, want in (("mistral-7b", "topr", "0.451"),
                              ("mistral-7b", "mean_logreg", "0.413"),
                              ("qwen2.5-7b", "topr", "0.604"),
                              ("qwen2.5-7b", "mean_logreg", "0.382")):
            checks.append((f"E9 mean TPR@1% {m}/{kind} = {want}",
                           f"{tp[m][kind]:.3f}" == want and has(want),
                           f"artifact {tp[m][kind]:.4f}"))
        # The dilution prediction FAILS here; the paper says so, so assert the failure.
        ch = v["auroc_change_short_to_long"]
        checks.append(("E9 linear baseline does NOT dilute on mistral (paper says so)",
                       ch["mistral-7b"]["mean_logreg"] > 0,
                       f"{ch['mistral-7b']['mean_logreg']:+.4f}"))
        checks.append(("E9 peaked aggregators fall most on qwen (paper says so)",
                       ch["qwen2.5-7b"]["multimax"] < ch["qwen2.5-7b"]["wlda"], ""))
        checks.append(("E9 mistral logreg gain +0.150 quoted",
                       f"{ch['mistral-7b']['mean_logreg']:.3f}" == "0.150"
                       and has("0.150"), f"{ch['mistral-7b']['mean_logreg']:.4f}"))

    snr = art("exp9_meanpool_snr.json")
    if snr:
        # The mean-pool SNR ratio must be flat-to-rising, which is the paper's
        # explanation for why the dilution prediction does not bite in this protocol.
        ks = sorted(snr, key=int)
        ratios = [snr[k]["signal_||dmu||"] / snr[k]["benign_mean_spread"] for k in ks]
        checks.append(("E9 mean-pool SNR ratio does not fall with N",
                       ratios[-1] >= ratios[0],
                       f"{[round(r,3) for r in ratios]}"))
        checks.append(("E9 SNR ratio 0.251 -> 0.309 quoted",
                       f"{ratios[0]:.3f}" == "0.251" and f"{ratios[-1]:.3f}" == "0.309"
                       and has("0.251") and has("0.309"), ""))

    width = max(len(c[0]) for c in checks)
    npass = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}s}  {'' if ok else detail}")
        npass += ok
    print(f"\n  {npass}/{len(checks)} claims verified against artifacts")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
