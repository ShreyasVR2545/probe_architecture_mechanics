# Benchmark notes

Hardware: NVIDIA RTX 5070 Laptop, 7.96 GiB VRAM. All figures batch size 1, d = 2048,
bf16 compute / fp32 reduction, chunk size 4096.

---

## Suite A — latency and memory vs context length

Peak memory is reported both raw and as **overhead above the input tensor**. The
`(1, N, 2048)` bf16 input is 512 MiB at N = 131,072 and dominates raw peak, which is why
a first run fitted every probe to α ≈ 0.55 and said nothing useful.

| probe | N=128 | N=1024 | N=8192 | N=32768 | N=131072 | overhead @131k | α (N≥8192) |
|---|---|---|---|---|---|---|---|
| `multimax` | 0.22 ms | 0.25 ms | 0.71 ms | 2.24 ms | **9.15 ms** | **35.1 MiB** | **0.000** |
| `mean_pool` | 0.21 | 0.24 | 0.71 | 2.24 | 9.31 | 35.1 MiB | 0.000 |
| `softmax_attn` | 0.24 | 0.27 | 0.75 | 3.25 | 12.69 | 652.1 MiB | 0.842 |
| `self_attn` | 0.27 | 0.30 | 8.99 | **936.97** | **OOM** | 10381.6 MiB | 1.960 |

**Read-off.**

- `multimax` and `mean_pool` are **O(1) in N** in their own footprint (α = 0.000, flat at
  35.1 MiB from N=8192 to N=131,072) because both stream the sequence in chunks and carry
  only a running reduction.
- `softmax_attn` is **O(N)** (α = 0.842): it materialises `(1, N, hidden)` to form the
  softmax-weighted sum.
- `self_attn` is **O(N²)** (α = 1.960), reaching 10.1 GiB at N = 32,768 and OOM-ing at
  131,072. Its latency at N = 32,768 is **937 ms vs 2.24 ms** for `multimax` — a 418×
  gap before it fails outright.

**Correction to the framing in the task brief.** Single-query attention pooling is
**O(N), not O(N²)** — its score tensor has shape `(1, N)`. The quadratic cost appears only
when the probe attends *across* the sequence. Both are measured here so the claim rests on
data. MultiMax's memory advantage over single-query pooling is the constant-in-N streaming
reduction (35 MiB vs 652 MiB at 131k), not an asymptotic class separation.

---

## Suite B — localized attack insertion

Attack: `k` consecutive tokens carrying a fixed direction scaled to **unit magnitude per
coordinate** (`√d` norm), strength 0.5 against background σ = 0.5. Probes trained **only**
at N = 512; N = 16,384 is strictly out of distribution. Recall at a threshold giving 1% FPR
on benign sequences of the **same length**.

| probe | k=4 | k=8 | k=16 | @train length |
|---|---|---|---|---|
| `multimax` | **1.000** | **1.000** | **1.000** | 1.000 |
| `softmax_attn` | 1.000 | 1.000 | 1.000 | 1.000 |
| `mean_pool` | 0.600 | 0.950 | 1.000 | 1.000 |

**The stated claim holds** — MultiMax keeps recall > 99% at every `k` under a 32× context
extension, dilution ratio k/N = 2.4×10⁻⁴.

**But the comparison at this strength is uninformative, and that is worth stating plainly:
`softmax_attn` also holds at 1.000.** Only `mean_pool` degrades (0.600 at k=4).

This is consistent with the theory rather than a refutation of it. Proposition 2.4 in
`math_formulation.tex` bounds the softmax mass on the attack by
`k·e^γ / (k·e^γ + N − k)`, which is O(1/N) only for **bounded** logit gap γ. Bounded is not
small: a well-separated needle lets the trained query reach `e^γ ~ N/k`, cancelling the
dilution exactly. The prediction is therefore a **crossover in signal strength**, not a
uniform MultiMax win. `benchmark_addendum.py` looked for it.

---

## Suite C — crossover sweep: **no MultiMax advantage over softmax pooling**

Fixed k=4, N=16,384 (dilution ratio 2.4×10⁻⁴), trained at N=512. Cells are
**recall@16384 / recall@512**; the second number separates "never learned the concept"
from "learned it, then diluted."

| strength | `multimax` | `softmax_attn` | `mean_pool` |
|---|---|---|---|
| 0.05 | 0.05 / 0.03 | 0.03 / 0.05 | 0.00 / 0.00 |
| 0.10 | **0.00 / 0.00** | **0.62 / 1.00** | 0.00 / 0.47 |
| 0.15 | 1.00 / 1.00 | 1.00 / 1.00 | 0.00 / 0.95 |
| 0.25 | 1.00 / 1.00 | 1.00 / 1.00 | 0.20 / 1.00 |
| 0.50 | 1.00 / 1.00 | 1.00 / 1.00 | 0.40 / 1.00 |

**Lowest strength holding recall ≥ 0.99 at N=16,384 (given the concept was learned at
training length): `multimax` 0.15, `softmax_attn` 0.15, `mean_pool` never.**

### This contradicts the premise, and the contradiction is the finding

The task brief frames MultiMax as beating "standard Softmax-Attention Probes." **These
measurements do not support that.** Both cross at exactly the same strength (0.15), and at
strength 0.10 MultiMax is strictly *worse*: it fails to learn at its own training length
(0.00) while softmax reaches 1.00 there and retains 0.62 under a 32× context extension.

The mechanism is gradient sparsity, now derived as Remark 2.8 in `math_formulation.tex`.
The subgradient of a hard max is supported on a **single position per head**, so each head
sees learning signal from one token per sequence, versus all N for softmax. Effective
sample size per step is H tokens rather than N. Near threshold — where the argmax has not
yet locked onto the attack span — that can stop the probe learning at all, independently of
any dilution effect.

**Corrected scope of the claim.** The demonstrated advantage of hard-max aggregation is
over **mean pooling**, whose Θ(1/N) decay has no free parameter to absorb it (mean_pool
never reaches 99% recall at any tested strength). It is **not** established over
single-query softmax pooling at N = 16,384. Any claim of MultiMax superiority over
attention pooling should be stated as conditional on the dilution ratio k/N and the
achievable logit gap γ — not as categorical.

What survives unambiguously is the **systems** result, not the statistical one: MultiMax
matches softmax's detection while using 35 MiB instead of 652 MiB of overhead at N=131k,
and 9.15 ms instead of 12.69 ms. That is a real deployment argument. Dilution resistance
relative to *attention* is not.

---

## Measurement bugs found and fixed

1. **Attack vector was unit-norm.** Spread over d = 2048 that is ~0.022 per coordinate
   against σ = 0.5 — far below the benign extremes a max reduction tracks. MultiMax scored
   **AUROC 0.484 at its own training length**, i.e. the benchmark was measuring noise. This
   is exactly the regime Remark 2.9 predicts (`E[max of N sub-Gaussians] = Θ(τ√(2 log N))`),
   so the module was behaving as derived. Fixed by √d-scaling.

2. **Peak memory is input-dominated**, collapsing all four scaling exponents to ≈ 0.55.
   Fixed by reporting overhead above the input and fitting on N ≥ 8192.

3. **Verdict dict dropped α = 0.000.** `{... for r in a if r.get("mem_scaling_alpha")}`
   uses truthiness, so exponents of exactly zero — the O(1) results the suite exists to
   find — were silently omitted from `benchmark_results.json` while appearing correctly in
   the console. Fixed to key on presence.

---

## Honest limitations

- **Synthetic hidden states.** Suites A and B run on `randn` backgrounds, not real model
  activations. This is deliberate — it makes N a free parameter on an 8 GiB card — but it
  means the attack geometry is idealised: a single fixed direction, additive, contiguous.
  Real misuse features are neither isolated nor axis-aligned.
- **The needle is contiguous and additive.** An adversary who spreads the signal across
  many low-magnitude tokens attacks MultiMax specifically, since the max sees only the
  single best position. That case is untested here.
- **Threshold drift is not measured over the full ladder.** Remark 2.9 predicts the benign
  maximum grows as √(2 log N), so a MultiMax threshold needs recalibrating with length.
  Suite B recalibrates per length by construction; the drift itself is not characterised.
- **`self_attn` is included as a scaling reference, not a serious baseline.** No one would
  deploy quadratic attention as a guardrail; it is there to show where the O(N²) claim
  actually applies.
