# Decisions register

One entry per judgement call, in the order taken. Predictions are written *before* the
measurement that tests them, so a wrong prediction stays on the record.

---

## D1. Streaming softmax pooling: predictions, written before running

**Date:** 2026-09-10 · **Stage:** 1 · **Status:** predictions logged, measurement pending

### The objection

The paper compares a *streamed* MultiMax against an *unstreamed* softmax pooling and
attributes the resulting 19.1 MiB vs 652.1 MiB gap to the aggregator. Single-query softmax
pooling admits an exact streaming form (online softmax, Milakov & Gimelshein 2018; the
mechanism behind FlashAttention, Dao et al. 2022) with carried state Θ(m):

    M   running max of the scores
    D   Σ_j exp(s_j − M)
    V   Σ_j exp(s_j − M) · y_j        ∈ R^m
    pooled = V / D

On inspection of `multimax_probe.py` the objection is structurally correct before any
measurement is taken:

* `SoftmaxAttnProbe.logits` materialises `y = (B, N, h)` and `w = (B, N)` over the whole
  sequence. There is no chunk loop. Its Θ(N) is a property of this implementation.
* `SelfAttnProbe.logits` builds `scores = (B, N, N)` explicitly rather than calling
  `F.scaled_dot_product_attention`. Its Θ(N²) is likewise implementation-dependent; the
  memory-efficient SDPA backend would make it Θ(N).

So the question is not *whether* the baseline can be streamed. It can. The question is how
much of the measured gap survives when it is.

### Predictions (before measurement)

| quantity | prediction | reasoning |
|---|---|---|
| streamed softmax overhead @131k, d=4096 | **≈ 8–16 MiB, flat** | carried state is Θ(m)=512 floats ≈ 2 KB, negligible; peak is set by the per-chunk transform output (C×m), which MultiMax also pays |
| streamed softmax α (N≥8192) | **≈ 0.000** | nothing scales with N once C < N |
| ratio streamed softmax : MultiMax | **1.0–2.0×** | same y buffer; softmax additionally holds scores (B,C) and weights (B,C), both O(C) not O(C·m), so a small constant |
| latency @131k | **within ~1.5× of MultiMax** | identical O(N·m·d) work in φ; the reduction differs by a rescale per chunk |
| trains successfully | **yes** | the recurrence is a composition of differentiable ops; nothing here is non-differentiable |
| detection parity with non-streamed | **exact to fp tolerance** | it computes the same function |
| self-attention with SDPA | **Θ(N), no OOM** | flash/mem-efficient backends never materialise N² |

**Therefore I predict Branch A: the memory gap largely collapses.** I expect the surviving
difference to be a constant factor (carried state Θ(H)=8 vs Θ(m)=512) that matters only in
the batched regime, not the 34× headline.

### What would falsify the prediction

* autograd retaining per-chunk activations so the *training* peak is Θ(N) for both, making
  the inference-only framing the real content
* the rescale introducing a serial dependency that costs enough latency to matter
* the per-chunk φ output dominating so completely that the reduction never mattered, in
  which case *both* arms are ~equal and the original comparison was measuring φ, not ρ
* numerical divergence in bf16 from the repeated rescaling

### Measurement (2026-09-10)

Numerical equivalence to the non-streamed form: **4.47e-08** max absolute deviation in
fp32, **exactly 0** in bf16, across N in {1024, 4096, 16384} and C in {256, 512, 1024,
4096}. The implementation computes the same function, so the comparison is valid.

| | streamed MultiMax | streamed softmax | non-streamed softmax |
|---|---|---|---|
| overhead @131k | 8.00 MiB | **12.04 MiB** | 641.0 MiB |
| alpha (N>=8192) | 0.0000 | **0.0000** | 0.9055 |
| latency @131k | 16.17 ms | 19.44 ms | 18.57 ms |
| trains | yes (AUROC 1.000) | yes (AUROC 1.000) | yes (AUROC 1.000) |

**The prediction was right and the objection is correct.** The 80x gap is 1.5x once the
baseline is streamed, and the exponent is 0.0000 for both.

### Where prediction and measurement diverged

Two places, both against my framing rather than for it.

1. I predicted the surviving 1.5x was the carried state, Theta(m)=512 floats against
   Theta(H)=8. It is not. 512 floats is 2 KB; the gap is 4 MiB. It comes from my
   streaming implementation upcasting the chunk's `y` to fp32 for the accumulation
   (4096 x 512 x 4 B = 8 MiB against 4 MiB in bf16). That is an accumulation-precision
   choice in my code, not a property of the aggregator. The honest reading is that the
   carried-state difference is **invisible at every scale measured**.

2. The Branch A brief anticipates that the Theta(H)-vs-Theta(m) constant "shows up in the
   batched regime". It does not. At N=65,536 the per-sequence overhead is **20.0 MiB for
   MultiMax, Top-r and streamed softmax alike** from B=2 to B=16. There is no batched
   constant-factor win to fall back on.

### Consequence if the prediction holds

§5.1's "this is what the paper rests on" is wrong as written and must be retracted, not
softened. The memory claim becomes a constant-factor claim plus a statement about what a
practitioner hits with a naive implementation. The paper's centre moves to Theorem 3.1,
the 1/N identity, and rank gating, all of which are independent of the reduction's memory
behaviour.
