"""Diagrams drawn from the code paths described in the Methods: study_overview.png and web_architecture_flow.png."""
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, "experiments/figures")
from figstyle import AQUA, BLUE, GREY, INK, INK2, ORANGE, save  # noqa: E402

FILL = {"data": "#eef4fc", "desk": "#fdf0ea", "web": "#e9f7f1", "rq": "#f4f3f0"}
EDGE = {"data": BLUE, "desk": ORANGE, "web": AQUA, "rq": GREY}


BOXES = []


def box(ax, x, y, w, h, text, kind, fs=6.9, bold_first=True):
    p = ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.08", facecolor=FILL[kind],
                                edgecolor=EDGE[kind], linewidth=0.9))
    lines = text.split("\n")
    if bold_first:
        t1 = ax.text(x + w / 2, y + h - 0.16, lines[0], ha="center", va="top", fontsize=fs, fontweight="bold", color=INK)
        ts = [t1]
        if len(lines) > 1:
            ts.append(ax.text(x + w / 2, y + h - 0.5, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs - 0.5,
                              color=INK2, linespacing=1.3))
    else:
        ts = [ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)]
    BOXES.append((p, ts))
    return (x, y, w, h)


def contained(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for p, ts in BOXES:
        pb = p.get_window_extent(r)
        for t in ts:
            tb = t.get_window_extent(r)
            if tb.x0 < pb.x0 + 2 or tb.x1 > pb.x1 - 2 or tb.y0 < pb.y0 + 1 or tb.y1 > pb.y1 - 1:
                bad.append(t.get_text().split("\n")[0])
    BOXES.clear()
    assert not bad, "text overflows its box: " + repr(bad)


def arrow(ax, p, q, text=None, off=(0, 0.12), color=INK2, fs=6.4, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=color,
                                 shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha=ha, va="bottom", fontsize=fs,
                color=INK2)


def canvas(w, h, W, H):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- study overview
fig, ax = canvas(6.8, 2.6, 13.6, 5.2)
box(ax, 0.1, 2.8, 2.7, 2.3, "Source clip\n191 frames\n1920\u00d71080 pixels\n29.97 frames s$^{-1}$\n3 packages", "data")
box(ax, 3.3, 2.8, 2.9, 2.3, "Roboflow export\n512\u00d7512 stretch\none class\nrandom frame split\n134 / 38 / 19 images", "data")
box(ax, 6.7, 2.8, 2.6, 2.3, "YOLOv12s\nUltralytics 8.4.160\n100 epochs\nbest epoch 98", "data")
box(ax, 9.8, 4.05, 3.7, 1.05, "Desktop (reference)\nPyTorch on every frame", "desk")
box(ax, 9.8, 2.8, 3.7, 1.05, "Web (deployed)\nJPEG q85, ONNX, sampler", "web")
arrow(ax, (2.8, 3.95), (3.3, 3.95))
arrow(ax, (6.2, 3.95), (6.7, 3.95))
arrow(ax, (9.3, 4.3), (9.8, 4.55))
arrow(ax, (9.3, 3.6), (9.8, 3.32))
box(ax, 3.3, 0.1, 2.9, 1.9, "RQ3 Validity\nleaky vs time-blocked\nretraining on one\ntest block", "rq")
box(ax, 6.7, 0.1, 2.6, 1.9, "RQ4 Explanation\ndeletion, insertion\npointing game\nweight randomisation", "rq")
box(ax, 9.8, 0.1, 1.8, 1.9, "RQ1 Parity\nraw and\nJPEG frames\nboxes, maps", "rq")
box(ax, 11.7, 0.1, 1.8, 1.9, "RQ2 Sampling\n4\u201315\nframes s$^{-1}$\ntwo rules", "rq")
arrow(ax, (4.75, 2.8), (4.75, 2.0))
arrow(ax, (8.0, 2.8), (8.0, 2.0))
arrow(ax, (10.7, 2.8), (10.7, 2.0))
arrow(ax, (12.6, 2.8), (12.6, 2.0))
contained(fig)
save(fig, "figures/study_overview.png")

# ---------------------------------------------------------------- web architecture
fig, ax = canvas(6.8, 3.4, 13.6, 6.8)
ax.add_patch(FancyBboxPatch((0.1, 0.1), 7.0, 6.55, boxstyle="round,pad=0,rounding_size=0.12", facecolor="white",
                            edgecolor=BLUE, linewidth=1.0, linestyle=(0, (4, 2))))
ax.text(0.35, 6.45, "Browser (all state)", fontsize=7.6, fontweight="bold", color=BLUE, va="top")
ax.add_patch(FancyBboxPatch((8.4, 1.9), 5.1, 4.75, boxstyle="round,pad=0,rounding_size=0.12", facecolor="white",
                            edgecolor=AQUA, linewidth=1.0, linestyle=(0, (4, 2))))
ax.text(8.65, 6.45, "Vercel Python function (stateless)", fontsize=7.6, fontweight="bold", color="#127a55", va="top")
box(ax, 0.4, 4.35, 3.0, 1.6, "Sampler\nseek to $t_k=k/f_s$\n$f_s\\in\\{4,6,10,15\\}$", "data")
box(ax, 3.8, 4.35, 3.0, 1.6, "Encoder\ndraw 1020\u00d7600\nJPEG quality 0.85", "data")
box(ax, 3.8, 2.3, 3.0, 1.6, "Tracker + rule\nIoU, then centre\nwindow or sign change", "data")
box(ax, 3.8, 0.35, 3.0, 1.6, "Crop + overlay\n12\u00d720 grid to frame\nmasked to the box", "data")
box(ax, 0.4, 0.35, 3.0, 3.55, "Dashboard\nlog, CSV, JSON\ncrops, ZIP\nbuilt in the\nbrowser", "data")
box(ax, 8.7, 4.25, 4.5, 1.7, "/api/detect\nletterbox 384\u00d7640, ONNX Runtime\nscore 0.25, NMS 0.7, grid", "web")
box(ax, 8.7, 2.2, 4.5, 1.6, "/api/analyze\nfixed two-question prompt\nflap open? damage?", "web")
box(ax, 8.7, 0.25, 4.5, 1.25, "Groq VLM (external)\nqwen/qwen3.8-27b", "rq")
arrow(ax, (3.4, 5.15), (3.8, 5.15))
arrow(ax, (6.8, 5.6), (8.7, 5.6))
ax.text(7.75, 5.72, "JPEG", ha="center", va="bottom", fontsize=6.3, color=INK2)
arrow(ax, (8.7, 4.55), (6.8, 3.55))
ax.text(7.75, 4.25, "boxes,\ngrid", ha="center", va="bottom", fontsize=6.3, color=INK2)
arrow(ax, (5.3, 2.3), (5.3, 1.95))
arrow(ax, (3.8, 3.1), (3.4, 3.1))
arrow(ax, (3.8, 1.15), (3.4, 1.15))
arrow(ax, (6.8, 1.15), (8.7, 2.5))
ax.text(7.3, 1.3, "crop", ha="center", va="top", fontsize=6.3, color=INK2)
arrow(ax, (10.95, 2.2), (10.95, 1.5))
contained(fig)
save(fig, "figures/web_architecture_flow.png")
