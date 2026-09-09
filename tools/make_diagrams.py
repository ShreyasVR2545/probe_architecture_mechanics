r"""make_diagrams.py -- the two structural diagrams for the reviewer response.

  figures/fig_arch_multimax.pdf    streaming MultiMax pipeline: chunk in, phi, running
                                   max carried across chunks, annealing and the STE clamp
  figures/fig_cascade_pipeline.pdf two-stage safety cascade with the Platt-calibrated gate

These are schematics, so nothing is read from an artifact; the only numbers shown are
structural constants of the design (chunk width, state shape) rather than measurements.
Every measured quantity lives in the result figures instead, where the claim checker can
reach it.

A note on the layout code, because it is the whole reason this file is not a pile of
magic numbers. Box sizes are MEASURED from the rendered text (`Canvas.fit_w` /
`Canvas.fit_h`), never estimated. Estimating from line count times font size was tried
first and every box holding a formula overflowed, because a line carrying \sum with
limits renders far taller and wider than its nominal point size. `Canvas.box` now refuses
outright to draw text wider than its frame, so an overflowing label is a loud failure at
generation time instead of a silent defect in the PDF.

Run:  python tools/make_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.5, "figure.dpi": 150,
    "pdf.fonttype": 3, "ps.fonttype": 3,          # see make_figures.py for why not 42
})

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#4A4A4A"
PALE = {"blue": "#DCEAF7", "orange": "#FBE3D6", "green": "#D9F0E6",
        "grey": "#ECECEC", "yellow": "#FCF3D4"}
LS = 1.45


class Canvas:
    def __init__(self, fig_w, fig_h, xlim, ylim):
        self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h))
        self.ax.set_xlim(0, xlim)
        self.ax.set_ylim(0, ylim)
        self.ax.axis("off")

    def _extent(self, text, fs, weight):
        t = self.ax.text(0, 0, text, ha="center", va="center", fontsize=fs,
                         weight=weight, linespacing=LS)
        self.fig.canvas.draw()
        bb = t.get_window_extent(renderer=self.fig.canvas.get_renderer())
        t.remove()
        return bb.transformed(self.ax.transData.inverted())

    def fit_h(self, text, fs, weight="normal", pad=0.24):
        return self._extent(text, fs, weight).height + 2 * pad

    def fit_w(self, text, fs, weight="normal", pad=0.28):
        return self._extent(text, fs, weight).width + 2 * pad

    def box(self, x, y, w, text, fc, ec, fs=8.4, weight="normal", h=None):
        h = self.fit_h(text, fs, weight) if h is None else h
        need = self.fit_w(text, fs, weight)
        if need > w + 1e-9:      # loud, because a silent overflow is exactly the bug
            raise ValueError(
                f"label needs {need:.2f} data units but the box is {w:.2f} wide; "
                f"shorten it, widen the box, or move the line to the caption.\n"
                f"  text: {text[:70]!r}")
        # clip_on=False on purpose. Patches clip at the axes limits by default while text
        # does not, so an overflowing row silently lost its FRAMES and kept its labels,
        # which looks like a styling bug rather than a layout overflow. Unclipped, any
        # overflow is plainly visible in the output.
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.025",
            facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=2, clip_on=False))
        self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                     fontsize=fs, zorder=3, weight=weight, linespacing=LS)
        return (x, y, w, h)

    def row(self, texts, y, fs, cols, x0, gap=0.26, weight="normal"):
        """Equal-width boxes laid out left to right from x0. Returns (xs, bw, bh, total).

        Left-anchored rather than centred inside a fixed panel width: centring inside a
        hand-chosen width silently pushed rows past the axes when the measured box width
        grew, and the canvas is deliberately oversized with bbox_inches="tight" cropping
        the slack, so there is nothing to centre against anyway.
        """
        bw = max(self.fit_w(t, fs, weight) for t in texts)
        bh = max(self.fit_h(t, fs, weight) for t in texts)
        xs = [x0 + i * (bw + gap) for i in range(len(texts))]
        for x, t, (fc, ec) in zip(xs, texts, cols):
            self.box(x, y, bw, t, fc, ec, fs=fs, weight=weight, h=bh)
        return xs, bw, bh, len(texts) * bw + (len(texts) - 1) * gap

    def arrow(self, p, q, color=GREY, lw=1.3, ls="-", rad=0.0):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                          linewidth=lw, color=color, linestyle=ls,
                                          connectionstyle=f"arc3,rad={rad}", zorder=1,
                                          clip_on=False))

    def save(self, name):
        # pad_inches 0.05; the LaTeX float spacing supplies the gap to the text.
        #
        # tight_layout() is deliberately NOT called here. These schematics have their
        # axes switched off and every box is sized by MEASURING rendered text and
        # converting to data units. tight_layout resizes the axes after that
        # measurement, so the data-to-display scale changes while the text stays at its
        # point size, and every label overflows its box. bbox_inches="tight" crops the
        # canvas without touching the axes, which is what these need.
        for ext in ("pdf", "png"):
            self.fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight",
                             pad_inches=0.05)
        plt.close(self.fig)
        print(f"  -> figures/{name}.pdf  +  .png")


# ======================================================================================
def fig_arch_multimax() -> None:
    """Streaming pipeline. The visual point is that the ONLY thing crossing a chunk
    boundary is the (B, H) running state, so nothing on the page grows with N."""
    c = Canvas(8.4, 6.0, 26.0, 8.4)      # oversized; bbox_inches="tight" crops the slack
    ax = c.ax
    X0 = 2.20                            # left edge of the content

    # ==== the long sequence, split into chunks ====
    ctxt = "\n" + r"$C=4096$ tokens"
    chunks = [r"chunk $c_1$" + ctxt, r"chunk $c_2$" + ctxt, r"$\cdots$" + ctxt,
              r"chunk $c_{\lceil N/C\rceil}$" + ctxt]
    grey4 = [(PALE["grey"], GREY)] * 4
    cxs, cw, ch, _ = c.row(chunks, 6.80, 8.2, grey4, X0)
    for i in range(3):
        c.arrow((cxs[i] + cw, 6.80 + ch / 2), (cxs[i + 1], 6.80 + ch / 2), lw=1.0)
    # Moved further left (1.85 rather than 1.30) and the feed arrow shortened to match,
    # so the label sits in its own column of whitespace instead of crowding chunk 1.
    ax.text(cxs[0] - 1.85, 6.80 + ch / 2, "residual\nstream " r"$x_{1..N}$",
            ha="center", va="center", fontsize=8.2, color=GREY, linespacing=LS)
    c.arrow((cxs[0] - 0.80, 6.80 + ch / 2), (cxs[0], 6.80 + ch / 2), lw=1.0)

    # ==== one chunk expanded ====
    stages = [
        "token transform\n" r"$y_j=\phi(x_j)$" "\n"
        r"$\mathbb{R}^d\!\rightarrow\!\mathbb{R}^m$",
        "head scores\n" r"$s_{j,h}=v_h^\top y_j$" "\n" r"$H$ heads",
        "chunk reduction\n" r"$\max_j$  (deploy)" "\n" r"$\mathrm{smax}_\tau$  (train)",
        "merge into state\n" r"$a_h\leftarrow\max(a_h,\cdot)$" "\n" "in place",
    ]
    cols = [(PALE["blue"], BLUE), (PALE["blue"], BLUE),
            (PALE["yellow"], ORANGE), (PALE["green"], GREEN)]
    # Lowered from 4.44 to 4.02 to open a clear band inside the top of the container for
    # the caption. Above the container the caption sat in the corridor the two dotted
    # expansion lines sweep through, and they ran across the words.
    row_y = 4.02
    xs, bw, hb, row_w = c.row(stages, row_y, 8.2, cols, X0 + 0.28)
    PX0, PW = X0, row_w + 0.56           # panel wraps the widest row
    ax.text(PX0 + PW / 2, 8.10, "Streaming MultiMax: activation overhead "
                                r"$\Theta(\min(C,N))$, independent of $N$ once $C<N$",
            ha="center", va="center", fontsize=10.5, weight="bold")

    # The caption is placed ABOVE the container rather than inside it. That is the
    # geometric fix: no white mask is needed because the text no longer shares space
    # with either the dashed border or the two dotted expansion lines, which now
    # terminate on the container's top corners below it.
    py0, py1 = 1.28, 6.10
    ax.plot([cxs[0], PX0], [6.72, py1], color=GREY, lw=0.8, ls=":", clip_on=False)
    ax.plot([cxs[0] + cw, PX0 + PW], [6.72, py1], color=GREY, lw=0.8, ls=":",
            clip_on=False)
    ax.add_patch(FancyBboxPatch((PX0, py0), PW, py1 - py0,
                                boxstyle="round,pad=0.01,rounding_size=0.03",
                                facecolor="white", edgecolor=GREY, linewidth=1.0,
                                linestyle="--", zorder=0, clip_on=False))
    # Inside the container, in the band opened above the stage row. The dotted lines
    # terminate on the container's top corners, above this, so nothing crosses it.
    ax.text(PX0 + 0.30, py1 - 0.34,
            "per-chunk work (repeats; allocates nothing that persists)",
            fontsize=8.0, color=GREY, style="italic", ha="left", va="center", zorder=4)

    for i in range(3):
        c.arrow((xs[i] + bw, row_y + hb / 2), (xs[i + 1], row_y + hb / 2))

    # the carried state: the whole argument of the paper in one box
    st = (r"carried state: $(B,H)$ running maxima $+$ bias" "\n"
          r"size does not depend on $N$")
    sw = c.fit_w(st, 8.2, "bold")
    sh = c.fit_h(st, 8.2, "bold")
    # The state box is dropped to 2.86 and the merge arrow lengthened accordingly, so the
    # arrowhead lands in clear space above the box rather than against its top edge.
    sx, sy = PX0 + PW - sw - 0.28, 2.48
    c.box(sx, sy, sw, st, PALE["green"], GREEN, fs=8.2, weight="bold", h=sh)
    c.arrow((xs[3] + bw / 2, row_y), (sx + sw * 0.78, sy + sh + 0.06), color=GREEN,
            lw=1.5)

    # Feedback routed BELOW the state box, never through it. The horizontal run is
    # shortened at both ends so the label below it is bounded by whitespace, not by the
    # dashes: this is why no background mask is needed here any more.
    y_fb = sy - 0.62
    x_ret = xs[0] + bw / 2
    c.arrow((sx + 0.55, sy), (sx + 0.55, y_fb), color=GREEN, lw=1.3, ls=(0, (4, 2)))
    c.arrow((sx + 0.55, y_fb), (x_ret, y_fb), color=GREEN, lw=1.3, ls=(0, (4, 2)))
    c.arrow((x_ret, y_fb), (x_ret, row_y), color=GREEN, lw=1.3, ls=(0, (4, 2)))
    ax.text((x_ret + sx + 0.55) / 2, y_fb - 0.20, "next chunk reuses the same buffer",
            fontsize=7.8, color=GREEN, ha="center", va="top", style="italic", zorder=5)

    # ==== the two training-time mechanisms ====
    # Two lines each. The explanatory third line these boxes used to carry now lives in
    # the LaTeX caption, which is where an explanation belongs and cannot overflow.
    t1 = ("annealed Boltzmann operator (training only)\n"
          r"$\mathrm{smax}_\tau(s)=\tau\log\left(\frac{1}{N}\sum_j e^{s_j/\tau}\right)$,"
          r"   $\tau\rightarrow 0$")
    t2 = ("straight-through clamp (bounded link)\n"
          r"forward $\Pi_{[-c,c]}(z)$,   backward $\equiv 1$")
    mech_cols = [(PALE["yellow"], ORANGE), (PALE["orange"], ORANGE)]
    # Placed BELOW the dashed container with a real gap, derived from the measured box
    # height rather than a fixed y. At a hard-coded y=0.05 these two boxes were tall
    # enough to punch through the container's bottom edge.
    mech_h = max(c.fit_h(t1, 8.0), c.fit_h(t2, 8.0))
    mech_y = py0 - mech_h - 0.42
    mxs, mw, mh, mrow_w = c.row([t1, t2], mech_y, 8.0, mech_cols,
                                PX0 + max(0.0, (PW - 2 * 5.0 - 0.45)) / 2, gap=0.45)
    c.arrow((xs[2] + bw / 2, row_y), (mxs[0] + mw * 0.55, mech_y + mh),
            color=ORANGE, lw=1.0, ls=(0, (3, 2)), rad=0.20)
    c.arrow((sx + sw, sy + sh / 2), (mxs[1] + mw * 0.55, mech_y + mh),
            color=ORANGE, lw=1.0, ls=(0, (3, 2)), rad=-0.20)

    c.save("fig_arch_multimax")


# ======================================================================================
def fig_cascade_pipeline() -> None:
    """Two-stage cascade. The visual point is that stage 2 is entered on a band around
    the calibrated decision boundary, not on the raw score."""
    c = Canvas(8.4, 4.6, 28.0, 6.8)      # oversized; tight bbox crops the slack
    ax = c.ax
    PX0 = 0.40

    # The gate is stated as a RANK band, not as |p - 1/2| < delta. Sections 6.3 measures
    # the probability-band version escalating exactly nothing at every budget up to 25%,
    # so a diagram showing it would now contradict the text.
    stages = [
        "request\n" r"$N$ up to $131{,}072$",
        "Stage 1: MultiMax probe\n" r"$\Theta(\min(C,N))$ memory" "\n"
        "one streaming pass",
        "calibration\n" r"$\hat p$ for the reported score" "\n"
        "(Platt or isotonic)",
        "rank gate\n" r"middle $\delta$-band of $\mathrm{rank}(z)$ ?",
    ]
    cols = [(PALE["grey"], GREY), (PALE["blue"], BLUE),
            (PALE["yellow"], ORANGE), (PALE["yellow"], ORANGE)]
    row_y = 3.20
    xs, bw, h, row_w = c.row(stages, row_y, 8.4, cols, PX0, gap=0.30)
    mid = row_y + h / 2
    for i in range(3):
        c.arrow((xs[i] + bw, mid), (xs[i + 1], mid))
    gate_r = xs[3] + bw

    t_ok = "confident:\ndecide now,\nno stage-2 cost"
    t_esc = "Stage 2:\nSGuard ContentFilter 2B\nfull forward pass"
    rw = max(c.fit_w(t_ok, 8.2), c.fit_w(t_esc, 8.2, "bold"))
    rx = gate_r + 1.05

    # Branch labels are separated GEOMETRICALLY, not masked. The connector gap is
    # widened to 1.05 and each label is anchored just inside the gate side of it, so the
    # label occupies clear canvas between the gate and its destination box. No white
    # background patch is used anywhere in this diagram.
    # Each label is offset PERPENDICULAR to its connector, on the side the arc bends
    # away from. A positive rad bows the curve up and left, so "no" goes below and right
    # of the chord midpoint; a negative rad bows down and left, so "yes" goes above and
    # right. Anchored near the arc's start the labels lay along the curve itself.
    hok = c.fit_h(t_ok, 8.2)
    ok_y = row_y + h + 0.46
    c.box(rx, ok_y, rw, t_ok, PALE["green"], GREEN, fs=8.2, h=hok)
    a_ok, b_ok = (gate_r, mid + h * 0.20), (rx, ok_y + hok / 2)
    c.arrow(a_ok, b_ok, color=GREEN, lw=1.4, rad=0.18)
    ax.text((a_ok[0] + b_ok[0]) / 2 + 0.16, (a_ok[1] + b_ok[1]) / 2 - 0.34, "no",
            fontsize=8.2, color=GREEN, weight="bold", ha="left", va="top", zorder=6)

    hesc = c.fit_h(t_esc, 8.2, "bold")
    esc_y = row_y - hesc - 0.50
    c.box(rx, esc_y, rw, t_esc, PALE["orange"], ORANGE, fs=8.2, weight="bold", h=hesc)
    a_es, b_es = (gate_r, mid - h * 0.20), (rx, esc_y + hesc / 2)
    c.arrow(a_es, b_es, color=ORANGE, lw=1.4, rad=-0.18)
    ax.text((a_es[0] + b_es[0]) / 2 + 0.16, (a_es[1] + b_es[1]) / 2 + 0.34, "yes",
            fontsize=8.2, color=ORANGE, weight="bold", ha="left", va="bottom", zorder=6)

    total_w = rx + rw - PX0
    ax.text(PX0 + total_w / 2, 6.52,
            "Two-stage safety cascade: a constant-memory gate in front of a "
            "heavyweight monitor",
            ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(PX0 + total_w / 2, 0.62,
            r"Gating a $\delta$-band of RANKS escalates exactly $\delta$ of traffic. "
            r"A band on $|\hat p-\frac{1}{2}|$ does not:"
            "\ncalibration compresses the score spread while preserving its order, and "
            "that gate escalates nothing below a 25% budget."
            "\nStage 2 runs on the escalated fraction only, so the compute saved is "
            r"$(1-\delta)$ of the stage-2 cost.",
            ha="center", va="center", fontsize=8.0, color=GREY, linespacing=1.6)

    c.save("fig_cascade_pipeline")


if __name__ == "__main__":
    fig_arch_multimax()
    fig_cascade_pipeline()
    print("diagrams done")
