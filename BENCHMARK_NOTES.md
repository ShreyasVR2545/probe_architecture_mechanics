# Benchmark notes

Hardware: NVIDIA RTX 5070 Laptop, 7.96 GiB VRAM. Batch size 1, d = 2048, bf16 compute /
fp32 reduction, chunk size 4096. All numbers from a single full run of
`benchmark_suite.py` (`logs/benchmark_full.log`, exit 0, no warnings under
`-W error::UserWarning`).

---

## Suite A — latency and memory vs context length

Peak memory is reported as **overhead above the input tensor**, because the
`(1, N, 2048)` bf16 input is 512 MiB at N = 131,072 and dominates raw peak.

| probe | 128 | 1,024 | 8,192 | 32,768 | 131,072 | overhead @131k | α |
|---|---|---|---|---|---|---|---|
| `multimax` | 0.253 ms | 0.263 | 0.707 | 2.293 | **9.205** | **19.1 MiB** | **0.000** |
| `mean_pool` | 0.288 | 0.482 | 0.667 | 2.392 | 9.438 | 35.1 MiB | 0.000 |
| `softmax_attn` | 0.238 | 0.267 | 0.720 | 3.125 | 12.729 | 652.1 MiB | 0.842 |
| `self_attn` | 0.279 | 0.373 | 9.182 | **1128.474** | **OOM** | 10381.6 MiB | 1.960 |

MultiMax overhead is flat at **19.1 MiB from N = 8,192 through 131,072** — a 16× length
increase at constant cost. `self_attn` is **492× slower** than MultiMax at N = 32,768 and
then OOMs, logged as a system limit rather than a crash.

**Correction to the brief's framing.** Single-query attention pooling is **O(N), not
O(N²)** — its score tensor is `(1, N)`. Only self-attention is quadratic.

---

## Suite B — contiguous needle, signal-strength sweep

k = 4 contiguous tokens in N = 16,384 (dilution 2.4×10⁻⁴), trained at N = 512. Cells are
**recall@16384 / recall@512**; the second number separates "never learned it" from
"learned it, then diluted". MultiMax trained with smooth-max annealing.

| strength | `multimax` | `softmax_attn` | `mean_pool` |
|---|---|---|---|
| 0.10 | **1.00 / 1.00** | 0.68 / 1.00 | 0.04 / 0.24 |
| 0.15 | 1.00 / 1.00 | 1.00 / 1.00 | 0.16 / 0.84 |
| 0.50 | 1.00 / 1.00 | 1.00 / 1.00 | 0.92 / 1.00 |

**Annealing fixes the failure found in the previous pass.** Without it MultiMax scored
0.00 / 0.00 at strength 0.10 — it never learned the concept, because the hard-max
subgradient reaches only H tokens per step. With annealing it reaches 1.00 / 1.00 and
**leads softmax (0.68)** in the weak-signal regime.

---

## Suite C — distributed / fragmented attack

Total signal budget fixed at S = 2.0, split across m non-contiguous tokens (per-token
S/m). Mean pooling sees an unchanged sum; a hard max sees a peak falling as 1/m.
N = 16,384.

| m | per-token | `multimax` | `softmax_attn` | `mean_pool` |
|---|---|---|---|---|
| 1 | 2.0000 | 1.000 | 1.000 | 0.140 |
| 4 | 0.5000 | 1.000 | 1.000 | 0.140 |
| 16 | 0.1250 | 1.000 | 1.000 | 0.140 |
| **64** | 0.0312 | **0.480** | **0.440** | 0.100 |
| 256 | 0.0078 | 0.020 | 0.000 | 0.060 |

**Failure boundary (recall < 0.80): `multimax` m=64, `softmax_attn` m=64, `mean_pool`
m=1.**

Two things worth stating precisely:

1. **The predicted MultiMax-specific vulnerability did not isolate.** Both pooled
   aggregators fail at the same m. A preliminary `--quick` run at N = 8,192 suggested
   softmax survived to m = 256; **that did not replicate at N = 16,384**, and the earlier
   figure is withdrawn.
2. **The boundary hides a real difference in ranking quality.** At m = 256 both have
   recall ≈ 0 at a 1% FPR threshold, but AUROC is **0.523 (multimax) vs 0.716
   (softmax)**. Softmax retains usable ordering after the thresholded detector has
   failed; the hard max loses its signal more completely once past the boundary.

---

## Calibration (Task 1.3)

| metric | before Platt | after Platt |
|---|---|---|
| Brier | 0.0338 | **0.0157** |
| NLL | 0.1351 | 0.0591 |
| accuracy | 0.963 | **0.981** |

Budget-driven δ: targets of 1 / 2 / 5 / 10% land at 1.2 / 2.5 / 5.0 / 10.0%.

---

## Bugs found and fixed by the self-healing loop

1. **`torch.clamp` causes zero-gradient trapping** — the exact failure the guard was
   specified to prevent. Replaced with `STEClamp(torch.autograd.Function)`. Verified at
   N = 131,072: `sum|param.grad| = 2.81e+05` (alive) vs `0.000e+00` for plain clamp.
2. **Unnormalised LogSumExp added `H·τ·log N` to the logit** — measured +42 at τ=1,
   N=512 and +68 at N=16,384. Large *and length-dependent*, so the probe relearned its
   bias each epoch and met a different offset at deployment length than at training
   length. Symptom: recall non-monotone in signal strength (1.00 at s=0.15, 0.04 at
   s=0.50). Fixed with the normalised Boltzmann operator `τ·log((1/N)Σexp(s/τ))`.
3. **Attack vector was unit-norm** (~0.022 per coordinate vs background σ = 0.5), making
   MultiMax score AUROC 0.484 at its own training length. Fixed by √d scaling.
4. **Peak memory is input-dominated**, collapsing all scaling exponents to ≈ 0.55. Fixed
   by reporting overhead above the input and fitting on N ≥ 8192.
5. **Verdict dict dropped α = 0.000** via a truthiness filter — precisely the O(1)
   results the suite exists to identify. Fixed to key on presence.
6. **Fixed δ = 0.05 escalated 0/160** on a well-calibrated probe. Correct behaviour, but
   it makes the gate untestable and silently degrades to probe-only. Replaced with
   `delta_for_escalation_rate()`, since deployments have an escalation budget.

---

## Honest limitations

- **Synthetic hidden states.** `randn` backgrounds with a single fixed additive attack
  direction. Real misuse features are neither isolated nor axis-aligned.
- **Suite C trains on the same m it tests.** The adversary's spread is known at training
  time, which is generous to the defender; an unknown-m attack is untested.
- **Threshold drift over length is not characterised.** Remark 2.9 predicts the benign
  maximum grows as √(2 log N); Suite B recalibrates per length by construction.
- **The LLM backend is simulated.** `SimulatedLLM` exercises routing and cost arithmetic
  only; it is not evidence about a real frontier model, and pricing constants are
  illustrative defaults rather than quoted prices.
- **LaTeX is structurally validated, not compiled** — no toolchain on this host.
