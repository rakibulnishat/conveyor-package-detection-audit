"""Shared style and checks for every manuscript figure.

* PNG only, 1000 dpi, white background.
* Width: 174 mm (double column) or 84 mm (single column) - Springer sn-jnl.
* Colours: categorical slots 1-3 of the validated reference palette (blue, orange, aqua),
  neutral greys for baselines; sequential maps are single-hue.
* check_overlaps(fig) raises if any two text boxes intersect, any text leaves the canvas,
  or a data point of a line/scatter falls inside an in-axes text box.
"""
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

MM = 1 / 25.4
W2 = 174 * MM
W1 = 84 * MM
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, GREY_L, INK, INK2 = "#7d7c78", "#c9c8c3", "#0b0b0b", "#52514e"
DPI = 1000

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": DPI, "figure.dpi": 150,
    "savefig.facecolor": "white", "axes.titleweight": "bold", "axes.titlelocation": "left",
    "mathtext.default": "regular",
})


def _hidden_ticklabels(fig):
    """Tick labels whose tick lies outside the view interval are not drawn; skip them."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tk.get_loc()
                if loc is None or not (lo - 1e-9 * abs(hi - lo) <= loc <= hi + 1e-9 * abs(hi - lo)):
                    hidden.add(id(tk.label1)); hidden.add(id(tk.label2))
    return hidden


def _texts(fig):
    out = []
    hidden = _hidden_ticklabels(fig)
    for t in fig.findobj(Text):
        if id(t) in hidden:
            continue
        if not t.get_visible() or not t.get_text().strip():
            continue
        out.append(t)
    return out


def check_overlaps(fig, pad=0.5, ignore_line_labels=()):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hidden = _hidden_ticklabels(fig)
    fb = fig.bbox
    texts = _texts(fig)
    boxes = [(t, t.get_window_extent(r).expanded(1.0, 1.0)) for t in texts]
    problems = []
    for t, b in boxes:
        if b.x0 < fb.x0 - 1 or b.y0 < fb.y0 - 1 or b.x1 > fb.x1 + 1 or b.y1 > fb.y1 + 1:
            problems.append(f"outside canvas: {t.get_text()!r}")
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ix > pad and iy > pad:
            problems.append(f"text overlap: {t1.get_text()!r} x {t2.get_text()!r}")
    # data marks under in-axes annotations
    for ax in fig.axes:
        leg = ax.get_legend()
        legbox = [(leg, leg.get_window_extent(r))] if leg is not None and leg.get_visible() else []
        inner = legbox + [(t, b) for t, b in boxes if t.axes is ax and t not in (ax.title, ax.xaxis.label, ax.yaxis.label)
                 and t not in ax.get_xticklabels() and t not in ax.get_yticklabels()
                 and t.get_text() not in ignore_line_labels]
        if not inner:
            continue
        pts = []
        for ln in ax.get_lines():
            if not ln.get_visible() or ln.get_label() == "_grid":
                continue
            xy = np.column_stack(ln.get_data()).astype(float)
            xy = xy[np.isfinite(xy).all(1)]
            if len(xy):
                D = ln.get_transform().transform(xy)
                if ln.get_linestyle() not in ("None", "", " ") and len(D) > 1:
                    dense = [D[:1]]
                    for p0, p1 in zip(D[:-1], D[1:]):
                        n = int(min(400, max(1, np.hypot(*(p1 - p0)) / 2)))
                        dense.append(p0 + (p1 - p0) * np.linspace(0, 1, n + 1)[1:, None])
                    D = np.vstack(dense)
                # keep only points inside the axes
                bb = ax.get_window_extent(r)
                D = D[(D[:, 0] >= bb.x0) & (D[:, 0] <= bb.x1) & (D[:, 1] >= bb.y0) & (D[:, 1] <= bb.y1)]
                pts.append(D)
        for col in ax.collections:
            off = col.get_offsets()
            if len(off):
                pts.append(col.get_offset_transform().transform(off))
        if not pts:
            continue
        P = np.vstack(pts)
        for t, b in inner:
            if not hasattr(t, "get_text"):
                b = b.expanded(0.97, 0.9)
            hit = (P[:, 0] > b.x0) & (P[:, 0] < b.x1) & (P[:, 1] > b.y0) & (P[:, 1] < b.y1)
            if hit.any():
                name = t.get_text() if hasattr(t, "get_text") else "legend"
                problems.append(f"data under label: {name!r} ({int(hit.sum())} pts)")
    return problems


PW, PH = 3.3, 2.45  # one panel, placed at width=.48\\columnwidth beside its partner


def panel(w=PW, h=PH):
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    return fig, ax


def save(fig, path, **kw):
    for ax in fig.axes:
        assert not ax.get_title(loc="left") and not ax.get_title() and not ax.get_title(loc="right"), \
            "no titles baked into panels: " + path
    probs = check_overlaps(fig, **kw)
    if probs:
        raise RuntimeError("overlap check failed for " + path + ":\n  " + "\n  ".join(probs))
    fig.savefig(path, dpi=DPI, format="png")
    plt.close(fig)
    print("saved", path)


def panel_label(ax, s):
    ax.set_title(s, loc="left")
