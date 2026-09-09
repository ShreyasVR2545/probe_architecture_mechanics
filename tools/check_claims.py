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

    width = max(len(c[0]) for c in checks)
    npass = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}s}  {'' if ok else detail}")
        npass += ok
    print(f"\n  {npass}/{len(checks)} claims verified against artifacts")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
