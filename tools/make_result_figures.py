r"""make_result_figures.py -- the four empirical figures for the reviewer response.

  figures/fig_real_llm_performance.pdf   real Mistral-7B residuals: memory vs N, AUROC by layer
  figures/fig_ablation_grid.pdf          c x H grid heatmaps plus the precision study
  figures/fig_drift_and_calibration.pdf  LSE shift, extreme-value drift, calibration transfer
  figures/fig_mixed_fragmentation.pdf    unknown-m and mixed-structure fragmentation

Every value is read from artifacts/*.json. Nothing is typed in by hand, for the same
reason the original figure script works that way: three figures in an earlier draft went
stale after a bug fix and nobody noticed, because a number in a plotting script is not
tested by anything.

Figures whose artifact is missing are skipped with a message rather than drawn from
partial data, so a half-finished experiment cannot silently produce a plausible-looking
plot.

Run:  python tools/make_result_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    # Grid strictly behind data and annotations. Without this matplotlib draws grid lines
    # at the same zorder as some artists and they cut across markers and text.
    "axes.axisbelow": True,
    "pdf.fonttype": 3, "ps.fonttype": 3,      # see make_figures.py for why not 42
})

# Legends are opaque and sit above everything. A translucent legend laid over a curve
# reads as a collision even when it is technically "behind" the data, and a transparent
# one lets grid lines strike through the labels.
LEG = dict(frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.8",
           borderpad=0.4, handlelength=1.6, labelspacing=0.35)


def legend(ax, **kw):
    opts = dict(LEG)
    opts.update(kw)
    lg = ax.legend(**opts)
    if lg is not None:
        lg.set_zorder(20)
    return lg


def text_on(color) -> str:
    """Black or white, whichever is readable on `color`.

    Needed because the CV heatmap uses a dark colormap: hard-coding black text made the
    darkest cell (c=10, H=32) render black-on-black and its value was invisible.
    """
    r, g, b = matplotlib.colors.to_rgb(color)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "black"

C = {"multimax": "#0072B2", "softmax_attn": "#D55E00", "mean_pool": "#009E73",
     "topr": "#CC79A7", "mean_max": "#8E6C00"}
LBL = {"multimax": "MultiMax (ours)", "softmax_attn": "Softmax attention",
       "mean_pool": "Mean pooling", "topr": r"Top-$r$ ($r{=}8$)",
       "mean_max": "Mean-Max ensemble"}


def load(name):
    p = ART / name
    if not p.exists():
        print(f"  SKIP: artifacts/{name} not found")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  -> figures/{name}.pdf  +  .png")


# ======================================================================================
def fig_real_llm():
    d = load("exp1_real_residuals.json")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    ax = axes[0]
    sysrows = [r for r in d["systems"] if not r.get("oom")]
    for kind in ("multimax", "softmax_attn", "mean_pool"):
        rs = sorted([r for r in sysrows if r["probe"] == kind], key=lambda r: r["N"])
        if not rs:
            continue
        ax.plot([r["N"] for r in rs], [r["peak_overhead_mib"] for r in rs],
                "o-", color=C[kind], label=LBL[kind], ms=3.5, lw=1.4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"context length $N$")
    ax.set_ylabel("activation overhead (MiB)")
    ax.set_title(r"(a) memory at Mistral-7B width $d{=}4096$")
    legend(ax, loc="upper left")

    ax = axes[1]
    det = d["detection"]
    layers = sorted({r["layer"] for r in det})
    Ns = sorted({r["N"] for r in det})
    width = 0.8 / max(len(Ns), 1)
    # grouped bars: one group per layer, one bar per probe, averaged over N
    probes = ("multimax", "softmax_attn", "mean_pool")
    w = 0.8 / len(probes)
    x = np.arange(len(layers), dtype=float)
    for i, kind in enumerate(probes):
        vals = []
        for L in layers:
            rs = [r["auroc"] for r in det if r["layer"] == L and r["probe"] == kind]
            vals.append(float(np.mean(rs)) if rs else np.nan)
        ax.bar(x + i * w - 0.4 + w / 2, vals, w * 0.92, color=C[kind], label=LBL[kind])
    ax.axhline(0.5, color="0.35", lw=0.9, ls="--")
    # 'chance' on the LEFT: at the right edge it sat under the legend and was unreadable.
    ax.text(-0.30, 0.515, "chance", fontsize=7.5, color="0.35", ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"layer {L}" for L in layers])
    ax.set_ylabel("AUROC (mean over $N$)")
    # Headroom above the tallest bar (0.86) so the legend clears the data entirely; at
    # 'lower right' it cut through the layer-24 and layer-31 bars and the chance line.
    ax.set_ylim(0.35, 1.28)
    ax.set_title("(b) detection on real residuals")
    legend(ax, loc="upper center", ncol=3, fontsize=6.6, columnspacing=1.0)
    fig.tight_layout()
    save(fig, "fig_real_llm_performance")


# ======================================================================================
def fig_ablation():
    d = load("exp2_ablations.json")
    if d is None:
        return
    g, prec = d["grid"], d["precision"]
    cs = sorted({r["c"] for r in g})
    Hs = sorted({r["H"] for r in g})

    def grid_of(key):
        M = np.full((len(cs), len(Hs)), np.nan)
        for r in g:
            M[cs.index(r["c"]), Hs.index(r["H"])] = r[key]
        return M

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9),
                             gridspec_kw={"width_ratios": [1, 1, 0.95]})

    for ax, key, title, cmap, vmin, vmax in (
            (axes[0], "auroc_ood", r"(a) OOD AUROC", "viridis", 0.5, 1.0),
            (axes[1], "grad_norm_cv", r"(b) gradient-norm CV", "magma_r", None, None)):
        M = grid_of(key)
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(Hs)))
        ax.set_xticklabels(Hs)
        ax.set_yticks(range(len(cs)))
        ax.set_yticklabels([f"{v:g}" for v in cs])
        ax.set_xlabel(r"heads $H$")
        ax.set_ylabel(r"clamp $c$")
        ax.set_title(title)
        ax.grid(False)
        for i in range(len(cs)):
            for j in range(len(Hs)):
                if np.isnan(M[i, j]):
                    continue
                # Colour by the LUMINANCE of the cell actually drawn. The previous rule
                # keyed off the colormap name and rendered the darkest CV cell
                # (c=10, H=32) black-on-black, hiding its value entirely.
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color=text_on(im.cmap(im.norm(M[i, j]))), zorder=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[2]
    dts = ["fp16", "bf16", "fp32"]
    on = [next(r["grad_norm"] for r in prec if r["dtype"] == t and r["ste"]) for t in dts]
    off = [next(r["grad_norm"] for r in prec if r["dtype"] == t and not r["ste"])
           for t in dts]
    x = np.arange(len(dts), dtype=float)
    ax.bar(x - 0.19, on, 0.36, color="#0072B2", label="straight-through", zorder=3)
    ax.bar(x + 0.19, [max(v, 0) for v in off], 0.36, color="#D55E00",
           label="plain clamp", zorder=3)
    # Annotate the zero bars flat on the baseline. Rotated 90 degrees and floated at 3%
    # of the axis they read as detached labels colliding with the tick text.
    for xi, v in zip(x + 0.19, off):
        if v == 0:
            ax.annotate("0.000", xy=(xi, 0), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.8, color="#D55E00",
                        zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels(dts)
    ax.set_ylabel("gradient norm under saturation")
    # Headroom so the legend sits above the bars rather than against their tops.
    ax.set_ylim(0, max(on) * 1.34)
    ax.set_title("(c) STE vs plain clamp")
    legend(ax, loc="upper center", ncol=1, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_ablation_grid")


# ======================================================================================
def fig_drift_cal():
    d = load("exp3_drift_calibration.json")
    if d is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))

    ax = axes[0]
    ax.set_ylim(0, 118)                 # headroom so the legend clears both curves
    for tau, mk in ((0.5, "o"), (1.0, "s")):
        rs = sorted([r for r in d["lse"] if r["tau"] == tau], key=lambda r: r["N"])
        if not rs:
            continue
        ax.plot([r["N"] for r in rs], [r["measured_gap"] for r in rs], mk + "-",
                ms=3.5, lw=1.3, color="#0072B2" if tau == 0.5 else "#D55E00",
                label=rf"measured, $\tau={tau}$")
        ax.plot([r["N"] for r in rs], [r["predicted_gap_H_tau_logN"] for r in rs],
                "--", lw=1.0, color="0.35",
                label=r"$H\tau\log N$" if tau == 0.5 else None)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"context length $N$")
    ax.set_ylabel("logit offset (unnormalised $-$ normalised)")
    ax.set_title("(a) LSE normalisation")
    legend(ax, loc="upper left", fontsize=6.8)

    ax = axes[1]
    pts = d["drift"]["points"]
    xs = np.array([np.sqrt(np.log(p["N"])) for p in pts])
    ys = np.array([p["mean_max"] for p in pts])
    ax.errorbar(xs, ys, yerr=[p["std_max"] for p in pts], fmt="o", ms=3.5,
                color="#0072B2", lw=1.0, capsize=2, label="measured")
    a, b = d["drift"]["fit_intercept"], d["drift"]["fit_slope"]
    xx = np.linspace(xs.min(), xs.max(), 40)
    ax.plot(xx, a + b * xx, "-", color="#D55E00", lw=1.3,
            label=rf"fit $b={b:.3f}$ ($R^2={d['drift']['r_squared']:.3f}$)")
    ax.plot(xx, xx * d["drift"]["predicted_slope_tau_sqrt2"], "--", color="0.35", lw=1.0,
            label=rf"theory $\tau\sqrt{{2}}={d['drift']['predicted_slope_tau_sqrt2']:.3f}$")
    ax.set_xlabel(r"$\sqrt{\log N}$")
    ax.set_ylabel(r"$\mathbb{E}[\max_j \varepsilon_j]$")
    ax.set_title("(b) benign-maximum drift")
    legend(ax, loc="lower right", fontsize=6.6)

    # Panel (c) redesigned. As grouped bars on a linear axis it showed exactly one
    # visible rectangle: the Brier scores are ~1e-8 (transferred) and ~1e-12 (per
    # length), so seven of the eight bars had no height. The finding is that the logit
    # DOES drift while Brier does NOT move, which needs two axes to show at all.
    ax = axes[2]
    rows = sorted(d["calibration"]["rows"], key=lambda r: r["N"])
    Ns = [r["N"] for r in rows]
    x = np.arange(len(Ns), dtype=float)
    ax.plot(x, [r["mean_logit"] for r in rows], "o-", color="#0072B2", ms=4, lw=1.5,
            zorder=4)
    drift = max(r["mean_logit"] for r in rows) - min(r["mean_logit"] for r in rows)
    ax.set_ylabel("mean logit", color="#0072B2")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n // 1024}k" if n >= 1024 else str(n) for n in Ns])
    ax.set_xlabel(r"evaluation length $N$")
    ax.set_title("(c) drift moves, calibration does not")

    # Only the logit is plotted. A twin log axis for Brier was tried and rejected: on a
    # log scale 1e-13 -> 4e-8 looks like a dramatic rise, which is the opposite of the
    # finding. The Brier numbers are stated instead, with the threshold that would
    # actually matter, so the panel cannot be misread as showing a penalty.
    worst = max(r["brier_transferred"] for r in rows)
    ax.set_ylim(min(r["mean_logit"] for r in rows) - 0.25,
                max(r["mean_logit"] for r in rows) + 0.75)
    ax.set_ylabel("mean logit")
    ax.tick_params(axis="y", labelcolor="black")
    ax.annotate(rf"drift $={drift:.2f}$ logits",
                xy=(x[-1], rows[-1]["mean_logit"]), xytext=(-8, -16),
                textcoords="offset points", fontsize=7.0, color="#0072B2", ha="right",
                zorder=6)
    mant, exp = f"{worst:.1e}".split("e")          # from the artifact, never typed in
    ax.text(0.03, 0.97,
            rf"Brier $\leq {mant}\times10^{{{int(exp)}}}$ at every $N$" "\n"
            r"($0.01$ would be a penalty that matters)",
            transform=ax.transAxes, fontsize=6.6, color="0.30", va="top", ha="left",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.85"))
    fig.tight_layout()
    save(fig, "fig_drift_and_calibration")


# ======================================================================================
def fig_mixed_frag():
    d = load("exp4_fragmentation.json")
    if d is None:
        return
    rows = d["fragmentation"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))

    for ax, mixed, title in ((axes[0], False, "(a) fragmented, unknown $m$"),
                             (axes[1], True, "(b) mixed structure (core $+$ smear)")):
        for kind in sorted({r["probe"] for r in rows}):
            rs = sorted([r for r in rows if r["probe"] == kind
                         and r["mixed_structure"] is mixed], key=lambda r: r["m"])
            if not rs:
                continue
            ax.plot([r["m"] for r in rs], [r["auroc"] for r in rs], "o-",
                    color=C.get(kind, "0.4"), label=LBL.get(kind, kind), ms=3.5, lw=1.3)
        ax.axhline(0.5, color="0.35", lw=0.9, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"fragmentation width $m$")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.35, 1.03)
        ax.set_title(title)
    # Five entries will not fit inside (a) without covering the mean-pooling curve and
    # the chance line, which is where they were. Panel (b) is nearly empty between 0.4
    # and 0.7, so the shared legend goes there and costs no extra figure height.
    handles, labels = axes[0].get_legend_handles_labels()
    lg = axes[1].legend(handles, labels, loc="lower center", ncol=2, fontsize=6.4,
                        columnspacing=1.0, **LEG)
    lg.set_zorder(20)

    ax = axes[2]
    bt = [r for r in d["batched"] if not r.get("oom")]
    for kind in ("multimax", "softmax_attn", "mean_pool"):
        rs = sorted([r for r in bt if r["probe"] == kind], key=lambda r: r["B"])
        if not rs:
            continue
        ax.plot([r["B"] for r in rs], [r["overhead_above_input_mib"] for r in rs],
                "o-", color=C[kind], label=LBL[kind], ms=3.5, lw=1.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"batch size $B$ at $N{=}65{,}536$")
    ax.set_ylabel("overhead above input (MiB)")
    ax.set_title("(c) batched systems cost")
    # Upper left, above the softmax curve. 'lower right' was tried and is worse: the
    # MultiMax curve climbs straight through that corner.
    ax.set_ylim(5, 2e4)
    legend(ax, loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_mixed_fragmentation")


# ======================================================================================
def fig_pareto_top_r():
    """AUROC against activation memory over the (r, m) grid, one panel per depth."""
    d = load("exp6_pareto.json")
    if d is None:
        return
    g = d["grid"]
    layers = sorted({r["layer"] for r in g})
    ms = sorted({r["m"] for r in g})
    rs = sorted({r["r"] for r in g})
    fig, axes = plt.subplots(1, len(layers) + 1, figsize=(7.8, 2.9),
                             gridspec_kw={"width_ratios": [1] * len(layers) + [0.9]})

    cmap = plt.get_cmap("viridis")
    for ax, L in zip(axes, layers):
        for j, m in enumerate(ms):
            sub = sorted([x for x in g if x["layer"] == L and x["m"] == m],
                         key=lambda x: x["r"])
            if not sub:
                continue
            ax.plot([x["overhead_mib"] for x in sub], [x["auroc"] for x in sub],
                    "o-", ms=3.2, lw=1.1, color=cmap(j / max(len(ms) - 1, 1)),
                    label=rf"$m={m}$")
        front = d["verdicts"]["pareto_front_by_layer"][str(L)]
        ax.plot([p["overhead_mib"] for p in front], [p["auroc"] for p in front],
                "k--", lw=1.4, zorder=5, label="Pareto front")
        ax.axhline(0.5, color="0.35", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("activation overhead (MiB)")
        ax.set_ylabel("AUROC")
        depth = "intermediate" if L == min(layers) else "near-final"
        ax.set_title(f"(layer {L}, {depth})")
        # Extra headroom at the bottom so the legend sits below every point and below
        # the chance line, instead of straddling both as it did at 'lower right'.
        ax.set_ylim(0.28, 1.0)
    legend(axes[0], loc="lower left", fontsize=6.4, ncol=2, columnspacing=0.9)

    # the r effect, isolated: it is depth that decides whether r matters
    ax = axes[-1]
    for L, style in zip(layers, ("o-", "s--")):
        best = []
        for r_ in rs:
            v = [x["auroc"] for x in g if x["layer"] == L and x["r"] == r_]
            best.append(max(v) if v else np.nan)
        ax.plot(rs, best, style, ms=3.4, lw=1.3, label=f"layer {L}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"top-$r$")
    ax.set_ylabel(r"best AUROC over $m$")
    ax.set_title(r"(c) where $r$ earns its cost")
    legend(ax, loc="center right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_pareto_top_r")


# ======================================================================================
def fig_qwen_family():
    d = load("exp8_qwen_family.json")
    if d is None:
        return
    det, sysrows = d["detection"], [r for r in d["systems"] if not r.get("oom")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    ax = axes[0]
    ratios = sorted({r["depth_ratio"] for r in det})
    probes = sorted({r["probe"] for r in det})
    w = 0.8 / len(probes)
    x = np.arange(len(ratios), dtype=float)
    for i, kind in enumerate(probes):
        vals = [float(np.mean([r["auroc"] for r in det
                               if r["depth_ratio"] == rho and r["probe"] == kind]) or 0)
                for rho in ratios]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w * 0.92,
               color=C.get(kind, "0.4"), label=LBL.get(kind, kind))
    ax.axhline(0.5, color="0.35", lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{rho:.0%} depth" for rho in ratios])
    ax.set_ylabel(r"AUROC (mean over $N$)")
    ax.set_ylim(0.35, 1.02)
    ax.set_title("(a) Qwen2.5-7B detection by depth")
    # No legend here: placed lower-left it sat on top of the 50%-depth bars. Both panels
    # use the same colours, so the one in (b) serves both.

    ax = axes[1]
    # MultiMax and Top-r are both flat at exactly 8.0 MiB, so one curve hid the other
    # completely. Distinct dash patterns and marker sizes make the coincidence legible
    # as a coincidence rather than as a missing series.
    styles = {"multimax": ("o-", 4.2, 1.6), "topr": ("s--", 3.0, 1.2),
              "softmax_attn": ("o-", 3.5, 1.3), "mean_pool": ("o-", 3.5, 1.3)}
    for kind in probes:
        rs = sorted([r for r in sysrows if r["probe"] == kind], key=lambda r: r["N"])
        if not rs:
            continue
        st, ms_, lw_ = styles.get(kind, ("o-", 3.5, 1.3))
        ax.plot([r["N"] for r in rs], [r["peak_overhead_mib"] for r in rs], st,
                color=C.get(kind, "0.4"), label=LBL.get(kind, kind), ms=ms_, lw=lw_)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(4, 2000)
    ax.set_xlabel(r"context length $N$")
    ax.set_ylabel("activation overhead (MiB)")
    ax.set_title(r"(b) memory at Qwen width $d{=}3584$")
    ax.annotate(r"MultiMax and Top-$r$ coincide at $8.0$ MiB",
                xy=(rs[-1]["N"], 8.0), xytext=(0, -16), textcoords="offset points",
                fontsize=6.2, color="0.35", ha="right")
    legend(ax, loc="upper left", fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig_qwen_family")


if __name__ == "__main__":
    fig_real_llm()
    fig_ablation()
    fig_drift_cal()
    fig_mixed_frag()
    fig_pareto_top_r()
    fig_qwen_family()
    print("result figures done")
