# Results

Model `HuggingFaceTB/SmolLM2-1.7B-Instruct`, SAE `EleutherAI/sae-SmolLM2-1.7B-layer17-32x`
(TopK, k=32, d_sae=65536, `resid_post` layer 17). 162 length-matched JBB-Behaviors
needles embedded in benign Pile filler at 0 → 7680 tokens. Peak VRAM 6.04 GiB.

---

## 1. Headline: the mechanism predicts the performance

Across the five probe architectures, **SNR retention correlates with long-context AUROC at
r = +0.982**.

| architecture | AUROC @0 | AUROC @7680 | decay | SNR @0 | SNR @7680 | SNR retention | positions used @7680 |
|---|---|---|---|---|---|---|---|
| `ema` | 0.794 | **0.789** | 0.005 | 0.681 | 0.604 | **0.887** | 0.00222 |
| `attn_hard` | 0.778 | 0.714 | 0.064 | 0.714 | 0.648 | **0.908** | 0.00013 |
| `linear_last` | 0.728 | 0.710 | 0.018 | 0.773 | 0.544 | 0.705 | 0.00013 |
| `mean_mlp` | 0.765 | 0.468 | 0.297 | 1.359 | 0.301 | **0.221** | 1.00000 |
| `attn_soft` | 0.790 | 0.472 | 0.319 | 1.152 | 0.245 | **0.212** | 0.86404 |
| *lexical (black-box)* | 0.706 | 0.521 | 0.184 | — | — | — | — |

SNR = |safety-latent score mass| / |background score mass|, computed by decomposing each
probe's score into per-latent contributions.

**The mechanism is pooling concentration, not capacity.** `mean_mlp` has 128× the
parameters of `ema` and decays 59× more. What separates them is how many token positions
their pooling admits: architectures that spread weight across all positions let the safety
latents be averaged into background as filler grows; architectures that concentrate weight
on <0.3% of positions hold their SNR.

**Softmax attention actively degrades with length.** Its pooling concentration goes
0.189 → 0.864 of positions used as context grows — the softmax *flattens* under more
tokens, so the gate that was supposed to select the needle instead dilutes it. The hard
argmax variant, identical except for the gate, holds at 0.00013 and retains 0.908 of its
SNR. This is the cleanest architectural contrast in the study.

**Operational consequence.** At 7680 filler tokens, `mean_mlp` and `attn_soft` are *worse
than a free TF-IDF baseline* (black-to-white boost −0.053 and −0.050). `ema` beats it by
+0.267. For a control-stack guardrail on long agent trajectories, the pooling rule is the
design decision that matters.

---

## 2. What the SAE did and did not earn

Reported as found, not tuned:

| claim | verdict |
|---|---|
| Score decomposition is faithful | **PASS** — Pearson r(true, reconstructed) = 0.89–0.97 |
| Safety latents beat a label-permutation null | **PASS**, but weakly — see limitation below |
| Readout alignment beats the 1/√d cosine null | **PASS** — max\|cos\| 0.104–0.195 vs p99 = 0.056 |
| SAE beats a random dictionary on max\|cos\| | **FAIL** — ratio 1.01–2.09 |
| SAE beats a random dictionary on top-64 span | **FAIL** — SAE 0.273–0.287 vs random 0.296–0.305 (**worse**) |
| H3-A support structure is SAE-specific | **PASS** — Jaccard 0.0968 (SAE) vs 0.0165 (random basis) vs 0.00049 (chance) |

The two failures replicate **Korznikov et al. (`2602.14111`)** in our own data: for
reconstructing a probe's readout *as a vector*, a random overcomplete dictionary of
identical shape does as well as the trained SAE, and on the span statistic slightly better.

The distinction that survives: the SAE earns its keep at the **activation level** — which
latents fire, on which examples, and how much score mass each contributes — not as a
**geometric basis** for probe weight vectors. The decisive control is the last row:
recomputing the cross-architecture overlap matrix in a random basis collapses it from
0.0968 to 0.0165, so the H3-A structure is a property of the SAE features, even though the
readout geometry is not.

---

## 3. Ensembles: heterogeneity helps only with skill weighting

Phase 4's first pass found heterogeneous ensembles **losing** to homogeneous ones
(−0.062 AUROC), apparently contradicting Koran et al. Phase 5 disentangled that from a
confound this setup introduces: under long-context shift the architectures are not equally
strong (solo AUROC spread = 0.266), and equal-weight z-score averaging gives a collapsed
member the same vote as a working one.

| aggregation | heterogeneous | homogeneous |
|---|---|---|
| equal weights (arithmetic mean) | 0.681 mean | 0.641 mean |
| **skill-weighted** | **0.760** | 0.662 |

Weighting each member by its solo skill above chance moves `het_broad` from 0.658 → 0.794
(+0.135) and `het_dilution_resistant` from 0.715 → 0.774 (+0.058).

**Refinement of Koran et al.:** diversity does pay under distribution shift, but only if
aggregation is skill-weighted. Arithmetic-mean ensembling — the aggregation their paper
used — is fragile precisely when members degrade unequally, which is what OOD shift causes.

**Error decorrelation (dim N=162, null p95 = 0.155).** Same architecture / different seeds:
mean error cosine 0.763. Different architectures: 0.499. Both sit far above the noise floor,
so members are not independent either way — but changing architecture decorrelates 1.53×
more than changing seed. Architecture, not initialisation, is the lever on ensemble
diversity.

---

## 4. H3-B: not supported, and underpowered

Does SAE feature-support overlap predict error correlation, so ensemble members could be
chosen without running the ensemble?

r = +0.370 across 10 architecture pairs; permutation test (20k shuffles) **p = 0.296**.
Direction of effect is as predicted, magnitude is not distinguishable from chance at n=10.
**Reported as underpowered, not refuted** — the design gives only C(5,2)=10 pairs, and
testing this properly needs more architectures or more layers.

---

## 5. Confounds handled

| confound | handling |
|---|---|
| **Sequence length** | First run measured raw needle length alone at AUROC **0.685** (train) / 0.615 (matched). Added greedy nearest-length pairing: **0.518 / 0.488**, and all six ladder rungs 0.483–0.492. |
| **Corpus source** | Probes hit ~1.00 in-distribution and ~0.99 on a held-out same-source split, but **0.73–0.79** on source-matched JBB — and the free lexical baseline drops identically (0.990 → 0.706). Much of in-distribution probe performance was corpus style. Measured, not hidden. |
| **Black-box sufficiency** | Every probe reported against a TF-IDF baseline (black-to-white boost). Two architectures fail it at long context. |
| **Formatting / needle identity** | Needle text byte-identical across every ladder rung; both classes get identical filler treatment. |
| **SAE off-distribution** | Decomposition encodes per-token residuals (in-distribution) and applies pooling weights afterwards, rather than encoding a pooled vector the SAE never saw. |
| **Base vs instruct dictionary** | SAE trained on base SmolLM2-1.7B; probing on -Instruct. Measured: FVU 0.088 vs 0.138 (Δ +0.050). |
| **Noise floor** | All cosines reported against empirical nulls: d=2048 (mean 0.0177, p95 0.0435) and N=162 (mean 0.0629, p95 0.1546). |

---

## 6. Limitations

1. **The safety-latent set is weakly certified.** Only **5 of 65,536** latents individually
   exceed the label-permutation threshold, which is a max-statistic null over all latents
   and therefore very conservative; the SNR analysis uses the top-128 by effect size, most
   of which do not individually clear that bar. The SNR result is internally consistent
   (r = +0.982 with AUROC, which pure noise would not produce) but a sensitivity sweep over
   `N_SAFETY` is the obvious next check and has not been run.
2. **One model, one layer, one SAE.** Layer 17 of a 1.7B model. Nothing here establishes
   scale behaviour, and the SAE is base-trained.
3. **n = 162 evaluation examples**, so AUROC differences below ~0.05 are not resolvable.
4. **`mean_mlp`'s readout direction is an expected input-gradient**, a local linearisation,
   not an exact weight vector. Its alignment numbers carry that caveat.
5. **Concept is misuse/sabotage-request detection**, not on-policy deception. The model is
   not lying; it is being asked to do something harmful inside an agent trajectory.
6. **The `MultiMax` attribution in the brief could not be verified** against
   `arXiv:2601.11516`; the architecture is implemented here as `AttnGatedProbe` with soft
   and hard gates on its own merits (see README).

---

## 7. Reproduce

```bash
python 01_environment_and_hooks.py          # ~2 min   env, hooks, SAE, nulls
python 02_probe_architectures.py            # ~12 min  train + OOD ladder
python 03_sae_mechanistic_decomposition.py  # ~12 min  SAE decomposition, SNR
python 04_ensemble_subspace_analysis.py     # ~15 min  ensembles, error subspaces
python 05_synthesis_and_figures.py          # ~5 s     synthesis (cache only)
```

Logs land in `logs/*.json`, the figure in `figures/summary.png`.
