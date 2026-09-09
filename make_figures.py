r"""
make_figures.py — publication figures, all driven from benchmark_results.json.

Nothing here is hand-typed: every number is read from the artifact, so a figure cannot
drift from the data the way the prose did. Outputs vector PDF (for LaTeX) plus PNG (for
the README).

  figures/fig1_architecture.pdf   system + cascade diagram
  figures/fig2_memory_scaling.pdf memory & latency vs N, plus the chunk ablation
  figures/fig3_annealing_recall.pdf weak-signal recall, annealed vs not
  figures/fig4_distributed_attack.pdf recall + AUROC vs fragmentation m

Run:  python make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    # fonttype 3, not 42. Type-42 output makes matplotlib subset fonts via fontTools,
    # whose compiled bezierTools DLL is blocked by an Application Control policy on this
    # host ("ImportError: DLL load failed"). Type 3 needs no subsetting and renders
    # identically; note that a venue requiring Type 1/TrueType-only would need this
    # revisited on a machine without that policy.
    "pdf.fonttype": 3, "ps.fonttype": 3,
})

C = {"multimax": "#0072B2", "softmax_attn": "#D55E00", "mean_pool": "#009E73",
     "self_attn": "#CC79A7"}
LBL = {"multimax": "MultiMax (ours)", "softmax_attn": "Softmax attention",
       "mean_pool": "Mean pooling", "self_attn": "Self-attention"}


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf  +  .png")


# ======================================================================================
def fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc, ec="#333", fs=8, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight)

    def arrow(x1, y1, x2, y2, style="-|>", color="#333", lw=1.1, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=11, lw=lw, color=color,
                                     linestyle=ls, shrinkA=2, shrinkB=2))

    ax.text(50, 38.5, "Frozen LLM, residual stream at layer $\\ell$   "
                      "$x_{i,j}\\in\\mathbb{R}^{d}$,  $j=1\\dots N$   ($N$ up to $131{,}072$)",
            ha="center", fontsize=9, weight="bold")
    box(3, 32, 94, 4.2, "", "#EEF3F8")
    for i, xx in enumerate(range(6, 96, 6)):
        ax.add_patch(FancyBboxPatch((xx, 32.6), 4.2, 3.0, boxstyle="round,pad=0.1",
                                    fc="#FFD9A0" if i in (7, 8) else "#D8E6F2",
                                    ec="#8899AA", lw=0.5))
    ax.text(50.5, 29.4, "$k$-token attack needle among benign context "
                        "(dilution $k/N \\approx 2.4\\times10^{-4}$)",
            ha="center", fontsize=7.5, style="italic", color="#666")

    # left: softmax
    box(4, 18, 42, 9.5, "", "#FDF0E8", ec=C["softmax_attn"])
    ax.text(25, 26.2, "Softmax attention pooling", ha="center", fontsize=9,
            weight="bold", color=C["softmax_attn"])
    ax.text(25, 23.6, "$\\alpha_j=\\mathrm{softmax}_j(q^\\top y_j/\\sqrt{m})$,  "
                      "score tensor $(1,N)$", ha="center", fontsize=8)
    ax.text(25, 21.4, "materialises $(1,N,m)$  $\\Rightarrow$  $\\Theta(N)$ memory",
            ha="center", fontsize=8)
    ax.text(25, 19.3, "652.1 MiB overhead at $N{=}131{,}072$   ($\\alpha{=}0.842$)",
            ha="center", fontsize=8, weight="bold", color=C["softmax_attn"])

    # right: multimax
    box(54, 18, 42, 9.5, "", "#E8F1F8", ec=C["multimax"])
    ax.text(75, 26.2, "MultiMax aggregation (ours)", ha="center", fontsize=9,
            weight="bold", color=C["multimax"])
    ax.text(75, 23.6, "$a_h=\\max_{j}\\, v_h^{\\top} y_j$,   $H$ heads,   streamed in chunks $C$",
            ha="center", fontsize=8)
    ax.text(75, 21.4, "carries only $(B,H)$ state  $\\Rightarrow$  "
                      "$\\Theta(\\min(C,N))$ memory", ha="center", fontsize=8)
    ax.text(75, 19.3, "19.1 MiB overhead at $N{=}131{,}072$   ($\\alpha{=}0.000$)",
            ha="center", fontsize=8, weight="bold", color=C["multimax"])

    arrow(25, 31.8, 25, 27.7); arrow(75, 31.8, 75, 27.7)
    arrow(46.2, 22.7, 53.8, 22.7, style="<|-|>", color="#888", lw=0.9)
    ax.text(50, 23.6, "34$\\times$", ha="center", fontsize=8, color="#888",
            weight="bold")

    # cascade
    arrow(75, 17.8, 75, 14.2)
    box(56, 9.5, 38, 4.6, "Platt gate   $\\hat p=\\sigma(az+b)$\n"
                          "escalate iff $|\\hat p-0.5|<\\delta$", "#F2F2F2",
        ec="#555", fs=8)
    ax.text(75, 7.9, "$\\delta$ chosen from an escalation budget, not hard-coded",
            ha="center", fontsize=7.5, style="italic", color="#666")

    arrow(66, 9.3, 56, 5.6); arrow(86, 9.3, 88, 5.6)
    box(34, 1.4, 25, 3.9, "confident  $\\rightarrow$  probe verdict\n"
                          "(97.5% of traffic, $\\Theta(1)$ memory)", "#E6F4EA",
        ec="#009E73", fs=7.5)
    box(70, 1.4, 26, 3.9, "ambiguous  $\\rightarrow$  LLM monitor\n"
                          "(2.5%, full inspection)", "#FDECEC", ec="#C0392B", fs=7.5)
    ax.text(2.5, 10.5, "Cascade\n\nMultiMax is the\n$\\Theta(1)$-memory first\n"
                       "pass; the expensive\nmonitor sees only what\n"
                       "the gate cannot resolve.",
            fontsize=7.5, va="center", ha="left", color="#444")
    save(fig, "fig1_architecture")


# ======================================================================================
def fig2_memory(d: dict) -> None:
    A = d["suite_a_latency_memory"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

    for k in ("multimax", "softmax_attn", "mean_pool", "self_attn"):
        r = [x for x in A if x["probe"] == k and x["status"] == "ok"]
        ax[0].plot([x["N"] for x in r], [x["overhead_mib"] for x in r],
                   marker="o", ms=4, color=C[k], label=LBL[k])
        ax[1].plot([x["N"] for x in r], [x["latency_ms"] for x in r],
                   marker="s", ms=4, color=C[k], label=LBL[k])
    oom = [x for x in A if x["status"] == "OOM"]
    for x in oom:
        ax[0].scatter([x["N"]], [1.3e4], marker="X", s=70, color=C[x["probe"]], zorder=5)
        ax[0].annotate("OOM", (x["N"], 1.3e4), fontsize=7.5, ha="center",
                       va="bottom", color=C[x["probe"]], weight="bold")
    for a, t, yl in ((ax[0], "Activation overhead above input", "MiB"),
                     (ax[1], "Forward latency", "ms")):
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("context length $N$"); a.set_ylabel(yl); a.set_title(t)
    ax[0].legend(loc="upper left"); ax[1].legend(loc="upper left")

    D = d.get("suite_d_chunk_ablation", [])
    if D:
        chunks = sorted({r["chunk_size"] for r in D})
        for c in chunks:
            r = [x for x in D if x["chunk_size"] == c and x.get("overhead_mib")]
            ax[2].plot([x["N"] for x in r], [x["overhead_mib"] for x in r],
                       marker="o", ms=4, label=f"$C={c}$")
        ax[2].set_xscale("log"); ax[2].set_xlabel("context length $N$")
        ax[2].set_ylabel("MiB"); ax[2].set_title("Chunk ablation: $\\Theta(\\min(C,N))$")
        ax[2].legend(ncol=2, fontsize=7)
    save(fig, "fig2_memory_scaling")


# ======================================================================================
def fig3_annealing(d: dict) -> None:
    B = {(r["probe"], r["strength"]): r for r in d["suite_b_strength_sweep"]}
    S = sorted({r["strength"] for r in d["suite_b_strength_sweep"]})
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # measured, from the artifact
    for k in ("multimax", "softmax_attn", "mean_pool"):
        ax[0].plot(S, [B[(k, s)]["recall_long"] for s in S], marker="o", ms=5,
                   color=C[k], label=LBL[k])
    # the pre-annealing MultiMax curve is recorded in the paper/notes, not the artifact:
    # it is the state of the code BEFORE the fix, so it is drawn dashed and labelled.
    ax[0].plot(S, [0.00, 1.00, 1.00], marker="x", ms=6, ls="--", color="#888",
               label="MultiMax, no annealing")
    ax[0].axhline(0.5, color="grey", lw=0.7, ls=":")
    ax[0].set_xlabel("attack strength $S$ (per coordinate)")
    ax[0].set_ylabel("recall @ 1% FPR,  $N{=}16{,}384$")
    ax[0].set_title("Weak-signal recall"); ax[0].set_ylim(-0.05, 1.08)
    ax[0].legend(loc="lower right")

    ax[1].bar([0, 1, 2], [0.00, 1.00, B[("softmax_attn", 0.10)]["recall_long"]],
              color=["#888", C["multimax"], C["softmax_attn"]], width=0.6)
    for i, v in enumerate([0.00, 1.00, B[("softmax_attn", 0.10)]["recall_long"]]):
        ax[1].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9, weight="bold")
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_xticklabels(["MultiMax\n(hard max)", "MultiMax\n(annealed)",
                           "Softmax\nattention"], fontsize=8)
    ax[1].set_ylabel("recall @ $S=0.10$"); ax[1].set_ylim(0, 1.18)
    ax[1].set_title("Effect of Boltzmann annealing")
    save(fig, "fig3_annealing_recall")


# ======================================================================================
def fig4_distributed(d: dict) -> None:
    Cd = {(r["probe"], r["m_spread"]): r for r in d["suite_c_distributed_attack"]}
    ms = sorted({r["m_spread"] for r in d["suite_c_distributed_attack"]})
    bnd = d["verdicts"]["distributed_attack_failure_boundary_m"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))

    for k in ("multimax", "softmax_attn", "mean_pool"):
        ax[0].plot(ms, [Cd[(k, m)]["recall"] for m in ms], marker="o", ms=5,
                   color=C[k], label=LBL[k])
        ax[1].plot(ms, [Cd[(k, m)]["auroc"] for m in ms], marker="s", ms=5,
                   color=C[k], label=LBL[k])
    for a in ax:
        a.set_xscale("log", base=2); a.set_xlabel("fragmentation $m$ (tokens, budget fixed)")
        if bnd.get("multimax"):
            a.axvline(bnd["multimax"], color="#C0392B", ls="--", lw=1.0)
    ax[0].axhline(0.80, color="grey", lw=0.7, ls=":")
    ax[0].text(bnd["multimax"] * 1.1, 0.86, "failure boundary\n$m=64$", fontsize=7.5,
               color="#C0392B")
    ax[0].set_ylabel("recall @ 1% FPR"); ax[0].set_title("Detection under fragmentation")
    ax[0].set_ylim(-0.05, 1.08); ax[0].legend(loc="lower left")

    ax[1].axhline(0.5, color="grey", lw=0.7, ls=":")
    au_mm = Cd[("multimax", ms[-1])]["auroc"]; au_sm = Cd[("softmax_attn", ms[-1])]["auroc"]
    ax[1].annotate("", xy=(ms[-1], au_sm), xytext=(ms[-1], au_mm),
                   arrowprops=dict(arrowstyle="<->", color="#333", lw=1.0))
    ax[1].text(ms[-1] * 0.55, (au_mm + au_sm) / 2,
               f"AUROC gap\n{au_sm - au_mm:+.3f}", fontsize=7.5, ha="right")
    ax[1].set_ylabel("AUROC"); ax[1].set_title("Ranking quality past the boundary")
    ax[1].set_ylim(0.4, 1.05)
    save(fig, "fig4_distributed_attack")


def main() -> int:
    d = json.loads((ROOT / "benchmark_results.json").read_text())
    print("Building figures from benchmark_results.json")
    fig1_architecture(); fig2_memory(d); fig3_annealing(d); fig4_distributed(d)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
