# Probe Architecture Mechanics

**Mechanistic Interpretability of Probe Architectures & Ensemble Subspaces under Distribution Shift**

Topics: `mechanistic-interpretability` · `ai-safety` · `sparse-autoencoders` · `probe-generalization` · `ai-control` · `pytorch`

> Note: "topics" are a GitHub-hosting concept, not native git metadata. They are recorded
> here and in `repo_metadata.json`; the repository description is set in `.git/description`.

---

## The question

White-box activation probes are now deployed safety infrastructure — shipped in
user-facing Gemini, and used as white-box members of untrusted-monitor ensembles in
Redwood-style AI control protocols. Two published findings sit unexplained next to each
other:

1. **Probe architecture drives OOD generalization** (Kramár et al., DeepMind): probes fail
   under production distribution shift, notably short → long context, and "a combination of
   architecture choice and training on diverse distributions is required for broad
   generalization." *Why* architecture matters is not explained.
2. **Diverse monitor ensembles beat homogeneous ones 2.4×** at equal compute (Koran et al.),
   and the best ensembles have low error-correlation between members. The paper explicitly
   does not investigate *why* diversity helps.

**Hypothesis (H3-A, primary).** Probe architecture determines *which features the probe
reads*. Last-token, mean-pooled, EMA and attention-gated probes trained on the same data for
the same concept converge on measurably different feature bases in an SAE dictionary.

**H3-B (exploratory).** Error-correlation between two probes is predicted by the overlap of
their SAE feature supports — so ensemble members can be selected from representational
overlap *before* running the ensemble.

**H3-C.** Architecture-driven feature differences explain generalization differences.

## The guardrail this repo is built around

This is **not** an AUROC bake-off. Every performance number must be accompanied by a
**mechanistic latent decomposition**: probe weight vectors are projected onto SAE decoder
directions, and the analysis reports *which sparse latents* each architecture reads, and
which ones get drowned in background residual noise as context grows.

---

## Hardware reality (why the model choice is what it is)

The brief specified a 16 GB budget, `< 11.5 GiB` peak, and Llama-3.1-8B or gemma-3-4b-it.
Measured on this machine:

| Fact | Value |
|---|---|
| GPU | NVIDIA RTX 5070 Laptop |
| VRAM total / free | **7.96 GiB / 6.83 GiB** |
| Llama-3.1-8B bf16 | ~16 GiB — does not fit |
| gemma-3-4b-it bf16 | ~8.6 GiB — exceeds total VRAM |
| All `google/gemma-*`, `meta-llama/*` | **403 GatedRepoError** on this account |

So both named models are excluded twice over: too large *and* licence-gated. The operative
ceiling is **7.0 GiB**, not 11.5.

**Chosen pairing — ungated and dimension-matched:**

| Component | Choice |
|---|---|
| Model | `HuggingFaceTB/SmolLM2-1.7B-Instruct` (24 layers, d=2048, 8192 ctx, bf16 ≈ 3.4 GiB) |
| SAE | `EleutherAI/sae-SmolLM2-1.7B-layer17-32x` (TopK, k=32, d_sae=65536, `resid_post` layer 17) |
| Hook layers | 11, 17, 21 (17 is the SAE site; 11/21 bracket it) |

Measured peak across all Phase-1 stages including an 8192-token forward: **4.883 GiB**.

**Known caveat, measured not assumed.** The SAE was trained on the *base* model
`HuggingFaceTB/SmolLM2-1.7B`; probing runs on *-Instruct*. Phase 1 quantifies the mismatch:

```
FVU on base      0.0884
FVU on instruct  0.1380      delta +0.0496     mean L0 = 32.0 both
```

The dictionary transfers. Both models remain available so a matched-base control can be run
for any Phase-3 claim that turns out to be sensitive to this.

If the Gemma licence is later accepted on Hugging Face, `src/common.py::Config` switches to
`gemma-3-1b-it` + `gemma-scope-2-1b-it` (an *instruct* SAE, no base/instruct mismatch at all)
by changing four fields — the SAE loader already handles the JumpReLU format.

---

## Statistical discipline (non-negotiable)

Carried from the proposal-stage critique passes:

- **Empirical `1/sqrt(d)` null.** In d=2048, two independent random unit vectors have
  mean |cos| = **0.0177** (analytic `1/sqrt(d)` = 0.0221), p95 = **0.0435**, p99 = 0.0561.
  Any reported cosine below p95 is indistinguishable from random. All vector overlaps are
  reported as percentiles against this null, never as raw numbers.
- **Split-half positive controls** on every direction estimate.
- **Causal claims outrank geometric ones** — orthogonal perturbations can be behaviourally
  equivalent, so geometry alone never establishes functional distinctness.
- **Random-dictionary control** on every SAE claim.
- **Black-to-white performance boost** as the headline metric: how much a white-box probe
  beats the black-box monitor it would replace. Beating chance is not a result.

## A citation caveat carried into this repo

The brief attributes **"MultiMax attention-gated probes"** to Kramár et al.
(`arXiv:2601.11516`). That paper's abstract describes novel probe architectures for
long-context generalization but **does not name MultiMax**, and an identical
"MultiMax / 88% → 3%" claim was traced to a search-engine summary and withdrawn during the
Pass-3 verification step. The architecture is well-defined and implemented here regardless,
but it is labelled **`AttnGatedProbe`** in code and described as *attention-gated pooling
with a hard-max variant*, not attributed to a specific paper's named method, pending a
full-text read of `2601.11516`.

---

## Layout

```
01_environment_and_hooks.py        Phase 1 — env, hooks, VRAM ceiling, SAE, null
02_probe_architectures.py          Phase 2 — 4 probe families, short->long OOD eval
03_sae_mechanistic_decomposition.py Phase 3 — probe -> SAE latent decomposition, SNR
04_ensemble_subspace_analysis.py   Phase 4 — ensemble error-vector subspace mechanics
src/common.py                      config, VRAM accounting, hooks, SAE loader, nulls
logs/                              JSON reports per phase (committed)
artifacts/                         derived tables (committed); raw tensors gitignored
figures/                           plots
```

## Reproducing

```bash
python 01_environment_and_hooks.py     # writes logs/01_environment_report.json
```

## Anchor papers

| # | Paper | Role here |
|---|---|---|
| 1 | Kramár, Engels, Wang, Chughtai, Shah, Nanda, Conmy — *Building Production-Ready Probes for Gemini* (`2601.11516`, DeepMind) | Probe architectures; short→long context OOD failure |
| 2 | Koran, Yun, Tetef, Arnav, Bernabeu-Pérez — *Ensemble Monitoring for AI Control* (`2605.15377`) | Heterogeneous vs homogeneous ensembles; error decorrelation |
| 3 | Natarajan, Jain, Arora, Golechha, Bloom — *One Probe Won't Catch Them All* (`2602.01425`, ICML 2026) | Deception is heterogeneous; 1D directions miss split clusters |
| 4 | Yoo & Skapars — *Probe Generalization as Subspace Selection* (`2609.02893`) | Principal-subspace overlap as the generalization mechanism |
| 5 | Greenblatt et al. — Redwood Research AI control corpus | Threat model: low-latency white-box guardrails in control stacks |

Supporting: Chanin et al. *A is for Absorption* (`2409.14507`); Korznikov et al.
*Sanity Checks for SAEs* (`2602.14111`); Cui et al. *SAE Interventions are Unreliable*
(`2606.18322`).
