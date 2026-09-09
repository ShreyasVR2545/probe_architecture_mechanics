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
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark_results.json"
CAL = ROOT / "logs" / "calibration_report.json"
NOTES = ROOT / "BENCHMARK_NOTES.md"
TEX = ROOT / "math_formulation.tex"


def approx(a: float, b: float, tol: float = 5e-3) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


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
