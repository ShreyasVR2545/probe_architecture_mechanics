r"""make_baseline_figures.py -- baseline comparison figures with bootstrap ribbons.

  docs/assets/fig_baseline_roc.{svg,png}      ROC curves, aggregators vs baselines
  docs/assets/fig_length_scaling.{svg,png}    AUROC and TPR@1%FPR vs N, 95% CI ribbons

Every value is read from artifacts/exp9_bootstrap_baselines.json.

Layout rules follow the house style and the brief: axes limits set explicitly, tick
padding at 6pt, legends outside the axes so nothing can be occluded, tight_layout plus
bbox_inches="tight" with pad_inches=0.05, and both a vector SVG and a 300 dpi PNG.

Run:  python tools/make_baseline_figures.py
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
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True,
    # Type 3 in PDF for the same Application Control reason as the other scripts; SVG
    # text is emitted as paths so the file needs no font resolution at all.
    "pdf.fonttype": 3, "ps.fonttype": 3, "svg.fonttype": "path",
})

COL = {"mean_logreg": "#8E6C00", "wlda": "#CC79A7", "latentbiopsy": "#56B4E9",
       "multimax": "#0072B2", "topr": "#009E73"}
LBL = {"mean_logreg": "Mean + logistic reg.", "wlda": "wLDA (Fisher)",
       "latentbiopsy": "LatentBiopsy (angular)", "multimax": r"MultiMax ($H{=}8$)",
       "topr": r"Top-$r$ ($r{=}8$)"}
ORDER = ["mean_logreg", "wlda", "latentbiopsy", "multimax", "topr"]


def load():
    p = ART / "exp9_bootstrap_baselines.json"
    if not p.exists():
        print("  SKIP: artifacts/exp9_bootstrap_baselines.json not found")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name):
    # SVG and 320 dpi PNG for the repository, PDF for the manuscript: pdflatex cannot
    # read SVG, so a vector PDF has to exist alongside it or the paper falls back to a
    # raster and prints soft.
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        kw = {"dpi": 320} if ext == "png" else {}
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)
    print(f"  -> docs/assets/{name}.svg + .png (320 dpi) + .pdf")


def tidy(ax, ylo=-0.02, yhi=1.02):
    ax.set_ylim(ylo, yhi)
    ax.tick_params(pad=6)


# ======================================================================================
def fig_roc(d):
    """ROC per model at the longest length, at each model's best-performing depth."""
    curves, rows = d["roc_curves"], d["rows"]
    models = sorted({r["model"] for r in rows})
    Nmax = max(r["N"] for r in rows)
    fig, axes = plt.subplots(1, len(models), figsize=(7.6, 3.1), squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, models):
        # pick the layer where the aggregators do best, so the baseline comparison is
        # made where the probe is actually usable rather than at an arbitrary depth
        cand = [r for r in rows if r["model"] == m and r["N"] == Nmax
                and r["aggregator"] == "topr"]
        layer = max(cand, key=lambda r: r["auroc"])["layer"] if cand else None
        for agg in ORDER:
            c = [c for c in curves if c["model"] == m and c["N"] == Nmax
                 and c["layer"] == layer and c["aggregator"] == agg]
            if not c:
                continue
            st = [r for r in rows if r["model"] == m and r["N"] == Nmax
                  and r["layer"] == layer and r["aggregator"] == agg][0]
            ax.plot(c[0]["fpr"], c[0]["tpr"], color=COL[agg], lw=1.5,
                    label=f"{LBL[agg]}  ({st['auroc']:.2f})")
        ax.plot([0, 1], [0, 1], color="0.55", lw=0.9, ls="--", zorder=1)
        ax.axvline(0.01, color="0.35", lw=0.8, ls=":", zorder=1)
        ax.set_xlim(-0.02, 1.02)
        tidy(ax)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(f"{m}, $N{{=}}{Nmax}$, layer {layer}")
    # One legend for both panels, outside the canvas on the right.
    h, l = axes[0].get_legend_handles_labels()
    lg = fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.005, 0.94),
                    frameon=True, fancybox=True, framealpha=0.8, borderpad=0.5)
    lg.get_frame().set_facecolor("white")
    lg.get_frame().set_edgecolor("#cccccc")
    save(fig, "fig_baseline_roc")


# ======================================================================================
def fig_length(d):
    """AUROC and TPR@1%FPR against N, averaged over depth, with 95% CI ribbons."""
    rows = d["rows"]
    models = sorted({r["model"] for r in rows})
    Ns = sorted({r["N"] for r in rows})
    fig, axes = plt.subplots(2, len(models), figsize=(7.8, 5.0), squeeze=False)

    specs = [("auroc", "auroc_lo", "auroc_hi", "AUROC"),
             ("tpr_at_1fpr", "tpr_lo", "tpr_hi", "TPR @ 1% FPR")]
    for col, m in enumerate(models):
        for row, (key, klo, khi, ylab) in enumerate(specs):
            ax = axes[row][col]
            for agg in ORDER:
                mu, lo, hi = [], [], []
                for N in Ns:
                    sel = [r for r in rows if r["model"] == m and r["N"] == N
                           and r["aggregator"] == agg]
                    if not sel:
                        mu.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                        continue
                    mu.append(float(np.mean([s[key] for s in sel])))
                    lo.append(float(np.mean([s[klo] for s in sel])))
                    hi.append(float(np.mean([s[khi] for s in sel])))
                ax.fill_between(Ns, lo, hi, color=COL[agg], alpha=0.14, lw=0)
                ax.plot(Ns, mu, "o-", color=COL[agg], ms=3.6, lw=1.5, label=LBL[agg])
            ax.axhline(0.5 if key == "auroc" else 0.0, color="0.5", lw=0.8, ls="--",
                       zorder=1)
            ax.set_xscale("log", base=2)
            tidy(ax)
            ax.set_xlabel(r"context length $N$")
            ax.set_ylabel(ylab)
            ax.set_title(f"{m}: {ylab} vs length")
    h, l = axes[0][0].get_legend_handles_labels()
    lg = fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.005, 0.92),
                    frameon=True, fancybox=True, framealpha=0.8, borderpad=0.5)
    lg.get_frame().set_facecolor("white")
    lg.get_frame().set_edgecolor("#cccccc")
    save(fig, "fig_length_scaling")


if __name__ == "__main__":
    data = load()
    if data is not None:
        fig_roc(data)
        fig_length(data)
        print("baseline figures done")
