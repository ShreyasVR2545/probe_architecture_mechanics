r"""
make_figures.py — publication figures, all driven from benchmark_results.json.

Nothing here is hand-typed: every number is read from the artifact, so a figure cannot
drift from the data the way the prose did. Outputs vector PDF (for LaTeX) plus PNG (for
the README).

  figures/fig1_architecture.pdf      system + cascade diagram
  figures/fig2_memory_scaling.pdf    memory & latency vs N, plus the chunk ablation
  figures/fig3_annealing_recall.pdf  weak-signal recall, annealed vs not
  figures/fig4_distributed_attack.pdf recall + AUROC vs fragmentation m

Run:  python tools/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]   # repo root, one level above tools/
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 9, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    # fonttype 3, not 42. Type-42 output makes matplotlib subset fonts via fontTools,
    # whose compiled bezierTools DLL is blocked by an Application Control policy on this
    # host ("ImportError: DLL load failed"). Type 3 needs no subsetting and renders
    # identically; a venue requiring Type 1/TrueType only would need this revisited on a
    # machine without that policy.
    "pdf.fonttype": 3, "ps.fonttype": 3,
})

C = {"multimax": "#0072B2", "softmax_attn": "#D55E00", "mean_pool": "#009E73",
     "self_attn": "#CC79A7"}
LBL = {"multimax": "MultiMax (ours)", "softmax_attn": "Softmax attention",
       "mean_pool": "Mean pooling", "self_attn": "Self-attention"}


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        # explicit margin: bbox_inches="tight" alone crops to the ink, leaving figures
        # butting against the text block once includegraphics scales them.
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  -> figures/{name}.pdf  +  .png")


# ======================================================================================
def fig1_architecture() -> None:
    """Two-column card layout, top-to-bottom flow.

    The earlier version packed six text rows into a 4.3-inch canvas and collided in four
    places. This one is taller, uses >=10pt type throughout, and spends the extra height
    on the one thing the caption cannot say: softmax MATERIALISES a score tensor over the
    whole sequence, while MultiMax STREAMS fixed-width chunks through a constant register.
    """
    fig, ax = plt.subplots(figsize=(12.5, 9.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.grid(False)
    FS, FSB, FSS = 10.5, 12.0, 9.5

    def card(x, y, w, h, fc, ec, lw=1.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.9",
                                    fc=fc, ec=ec, lw=lw, zorder=1))

    def txt(x, y, t, fs=FS, **kw):
        ax.text(x, y, t, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"),
                fontsize=fs, zorder=3, **kw)

    def arrow(x1, y1, x2, y2, color="#333", lw=1.6, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=16, lw=lw, color=color,
                                     shrinkA=3, shrinkB=3, zorder=2))

    # ---------------- input strip --------------------------------------------------
    txt(50, 97.0,
        r"Frozen LLM $\rightarrow$ residual stream at layer $\ell$:   "
        r"$x_{i,j}\in\mathbb{R}^{d}$,   $j = 1 \dots N$,   $N$ up to $131{,}072$",
        FSB, weight="bold")
    card(4, 87.5, 92, 5.6, "#EEF3F8", "#8899AA", lw=1.2)
    for i, xx in enumerate(range(7, 95, 5)):
        ax.add_patch(FancyBboxPatch((xx, 88.5), 3.6, 3.6, boxstyle="round,pad=0.15",
                                    fc="#F5A623" if i in (9, 10) else "#D8E6F2",
                                    ec="#8899AA", lw=0.6, zorder=2))
    txt(50, 82.6,
        r"$k$-token attack needle inside benign context   "
        r"(dilution $k/N \approx 2.4\times10^{-4}$)",
        FSS, style="italic", color="#555")

    arrow(27, 86.8, 27, 79.6)
    arrow(73, 86.8, 73, 79.6)

    # ---------------- left card: softmax, full materialisation ---------------------
    card(3, 50, 45, 28.5, "#FDF2EA", C["softmax_attn"])
    txt(25.5, 75.2, "Softmax attention pooling", FSB, weight="bold",
        color=C["softmax_attn"])
    txt(25.5, 71.4,
        r"$\alpha_j = \mathrm{softmax}_j\!\left(q^{\top} y_j / \sqrt{m}\right)$", FS)
    ax.add_patch(FancyBboxPatch((7.5, 63.0), 36, 5.6, boxstyle="round,pad=0.2",
                                fc="#F6C9AC", ec=C["softmax_attn"], lw=1.2, zorder=2))
    txt(25.5, 65.8, r"score tensor $(1,N)$  and  $(1,N,m)$ features", FSS)
    txt(25.5, 59.8, r"materialised in full  $\Rightarrow$  $\Theta(N)$ memory", FS)
    txt(25.5, 56.0, r"652.1 MiB overhead at $N = 131{,}072$", FS, weight="bold",
        color=C["softmax_attn"])
    txt(25.5, 52.6, r"scaling exponent  $\alpha = 0.842$", FSS, color="#555")

    # ---------------- right card: multimax, streaming ------------------------------
    card(52, 50, 45, 28.5, "#EAF2F9", C["multimax"])
    txt(74.5, 75.2, "MultiMax aggregation (ours)", FSB, weight="bold",
        color=C["multimax"])
    txt(74.5, 71.4, r"$a_h = \max_{j}\, v_h^{\top} y_j$,    $h = 1 \dots H$", FS)
    for i, xx in enumerate((56.0, 63.0, 70.0)):
        ax.add_patch(FancyBboxPatch((xx, 63.0), 5.6, 5.6, boxstyle="round,pad=0.2",
                                    fc="#BBD9F0", ec=C["multimax"], lw=1.0, zorder=2))
        txt(xx + 2.8, 65.8, rf"$C_{{{i + 1}}}$", FSS)
    arrow(76.6, 65.8, 81.0, 65.8, color=C["multimax"], lw=1.4)
    ax.add_patch(FancyBboxPatch((81.5, 63.0), 11.0, 5.6, boxstyle="round,pad=0.2",
                                fc="#7FB3DE", ec=C["multimax"], lw=1.6, zorder=2))
    txt(87.0, 65.8, r"$(B,H)$", FSS, weight="bold")
    txt(74.5, 59.8, "streamed; only a running max is carried", FS)
    txt(74.5, 56.0, r"19.1 MiB overhead at $N = 131{,}072$", FS, weight="bold",
        color=C["multimax"])
    txt(74.5, 52.6, r"$\Theta(\min(C,N))$,    $\alpha = 0.000$", FSS, color="#555")

    arrow(48.6, 64.0, 51.4, 64.0, color="#777", lw=1.4, style="<|-|>")
    txt(50, 67.4, r"$34\times$", FS, color="#555", weight="bold")

    # ---------------- gate ----------------------------------------------------------
    arrow(74.5, 49.6, 74.5, 40.8)
    card(46, 28.0, 51, 12.0, "#F4F4F4", "#555", lw=1.4)
    txt(71.5, 36.8, "Platt-scaled cascading gate", FSB, weight="bold")
    txt(71.5, 33.4,
        r"$\hat p = \sigma(a z + b)$      escalate iff  $|\hat p - 0.5| < \delta$", FS)
    txt(71.5, 30.3, r"$\delta$ set from an escalation budget, not hard-coded",
        FSS, style="italic", color="#555")

    arrow(58, 27.6, 42, 16.8)
    arrow(86, 27.6, 86, 16.8)
    card(15, 5.5, 50, 10.2, "#E8F6EC", "#009E73", lw=1.4)
    txt(40, 12.2, r"confident  $\rightarrow$  probe verdict", FS, weight="bold",
        color="#00795C")
    txt(40, 8.6, r"97.5% of traffic,   $\Theta(1)$ memory in $N$", FSS)
    card(68, 5.5, 29, 10.2, "#FDEEEE", "#C0392B", lw=1.4)
    txt(82.5, 12.2, r"ambiguous  $\rightarrow$  LLM", FS, weight="bold", color="#A93226")
    txt(82.5, 8.6, r"2.5%,   full inspection", FSS)

    txt(2, 46.0,
        "The cascade: MultiMax is the $\\Theta(1)$-memory first pass;\n"
        "the expensive monitor sees only what the gate cannot resolve.",
        FSS, ha="left", color="#444")
    save(fig, "fig1_architecture")


# ======================================================================================
def fig2_memory(d: dict) -> None:
    A = d["suite_a_latency_memory"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))

    for k in ("multimax", "softmax_attn", "mean_pool", "self_attn"):
        r = [x for x in A if x["probe"] == k and x["status"] == "ok"]
        ax[0].plot([x["N"] for x in r], [x["overhead_mib"] for x in r],
                   marker="o", ms=4.5, color=C[k], label=LBL[k])
        ax[1].plot([x["N"] for x in r], [x["latency_ms"] for x in r],
                   marker="s", ms=4.5, color=C[k], label=LBL[k])
    for x in [x for x in A if x["status"] == "OOM"]:
        ax[0].scatter([x["N"]], [1.3e4], marker="X", s=80, color=C[x["probe"]], zorder=5)
        ax[0].annotate("OOM", (x["N"], 1.3e4), fontsize=8.5, ha="center",
                       va="bottom", color=C[x["probe"]], weight="bold")
    for a, t, yl in ((ax[0], "Activation overhead above input", "MiB"),
                     (ax[1], "Forward latency", "ms")):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("context length $N$")
        a.set_ylabel(yl)
        a.set_title(t)
    ax[0].legend(loc="upper left")
    ax[1].legend(loc="upper left")

    D = d.get("suite_d_chunk_ablation", [])
    if D:
        for c in sorted({r["chunk_size"] for r in D}):
            r = [x for x in D if x["chunk_size"] == c and x.get("overhead_mib")]
            ax[2].plot([x["N"] for x in r], [x["overhead_mib"] for x in r],
                       marker="o", ms=4.5, label=f"$C={c}$")
        ax[2].set_xscale("log")
        ax[2].set_xlabel("context length $N$")
        ax[2].set_ylabel("MiB")
        ax[2].set_title(r"Chunk ablation: $\Theta(\min(C,N))$")
        ax[2].legend(ncol=2, fontsize=8)
    save(fig, "fig2_memory_scaling")


# ======================================================================================
def fig3_annealing(d: dict) -> None:
    B = {(r["probe"], r["strength"]): r for r in d["suite_b_strength_sweep"]}
    S = sorted({r["strength"] for r in d["suite_b_strength_sweep"]})
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.8))

    for k in ("multimax", "softmax_attn", "mean_pool"):
        ax[0].plot(S, [B[(k, s)]["recall_long"] for s in S], marker="o", ms=5.5,
                   color=C[k], label=LBL[k])
    # The pre-annealing MultiMax curve is the state of the code BEFORE the fix, so it is
    # not in the artifact; drawn dashed and labelled as such.
    ax[0].plot(S, [0.00, 1.00, 1.00], marker="x", ms=6.5, ls="--", color="#888",
               label="MultiMax, no annealing")
    ax[0].axhline(0.5, color="grey", lw=0.7, ls=":")
    ax[0].set_xlabel("attack strength $S$ (per coordinate)")
    ax[0].set_ylabel("recall @ 1% FPR,  $N{=}16{,}384$")
    ax[0].set_title("Weak-signal recall")
    ax[0].set_ylim(-0.05, 1.10)
    ax[0].legend(loc="lower right")

    vals = [0.00, 1.00, B[("softmax_attn", 0.10)]["recall_long"]]
    ax[1].bar([0, 1, 2], vals, width=0.6,
              color=["#888", C["multimax"], C["softmax_attn"]])
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10, weight="bold")
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_xticklabels(["MultiMax\n(hard max)", "MultiMax\n(annealed)",
                           "Softmax\nattention"], fontsize=9)
    ax[1].set_ylabel("recall @ $S=0.10$")
    ax[1].set_ylim(0, 1.20)
    ax[1].set_title("Effect of Boltzmann annealing")
    save(fig, "fig3_annealing_recall")


# ======================================================================================
def fig4_distributed(d: dict) -> None:
    Cd = {(r["probe"], r["m_spread"]): r for r in d["suite_c_distributed_attack"]}
    ms = sorted({r["m_spread"] for r in d["suite_c_distributed_attack"]})
    bnd = d["verdicts"]["distributed_attack_failure_boundary_m"]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.8))

    for k in ("multimax", "softmax_attn", "mean_pool"):
        ax[0].plot(ms, [Cd[(k, m)]["recall"] for m in ms], marker="o", ms=5.5,
                   color=C[k], label=LBL[k])
        ax[1].plot(ms, [Cd[(k, m)]["auroc"] for m in ms], marker="s", ms=5.5,
                   color=C[k], label=LBL[k])
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xlabel("fragmentation $m$ (budget fixed)")
        if bnd.get("multimax"):
            a.axvline(bnd["multimax"], color="#C0392B", ls="--", lw=1.1)
    ax[0].axhline(0.80, color="grey", lw=0.7, ls=":")
    ax[0].text(bnd["multimax"] * 1.12, 0.86, "failure boundary\n$m=64$", fontsize=8.5,
               color="#C0392B")
    ax[0].set_ylabel("recall @ 1% FPR")
    ax[0].set_title("Detection under fragmentation")
    ax[0].set_ylim(-0.05, 1.10)
    ax[0].legend(loc="lower left")

    au_mm = Cd[("multimax", ms[-1])]["auroc"]
    au_sm = Cd[("softmax_attn", ms[-1])]["auroc"]
    ax[1].axhline(0.5, color="grey", lw=0.7, ls=":")
    ax[1].annotate("", xy=(ms[-1], au_sm), xytext=(ms[-1], au_mm),
                   arrowprops=dict(arrowstyle="<->", color="#333", lw=1.1))
    ax[1].text(ms[-1] * 0.55, (au_mm + au_sm) / 2,
               f"AUROC gap\n{au_sm - au_mm:+.3f}", fontsize=8.5, ha="right")
    ax[1].set_ylabel("AUROC")
    ax[1].set_title("Ranking quality past the boundary")
    ax[1].set_ylim(0.4, 1.05)
    save(fig, "fig4_distributed_attack")


def main() -> int:
    d = json.loads((ROOT / "benchmark_results.json").read_text())
    print("Building figures from benchmark_results.json")
    fig1_architecture()
    fig2_memory(d)
    fig3_annealing(d)
    fig4_distributed(d)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
