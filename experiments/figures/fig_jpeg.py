"""JPEG quality sweep: agreement of the ONNX path with the PyTorch reference (raw frames) per quality.
jpeg_boxes.png (a), jpeg_cam.png (b)."""
import json
import sys

import numpy as np

sys.path.insert(0, "experiments/figures")
from figstyle import BLUE, GREY, INK2, ORANGE, panel, save  # noqa: E402

R = json.load(open("experiments/results/jpeg/jpeg_tradeoff.json"))
q = [r["quality"] for r in R]
xs = np.arange(len(q))
lab = [f"{r['quality']}\n{r['kb_median']:.0f}" for r in R]
dep = q.index(85)

fig, a = panel()
a.plot(xs, [1 - r["iou_median"] for r in R], color=BLUE, marker="o", ms=4, lw=1.2, label="Median pair")
a.plot(xs, [1 - r["iou_min"] for r in R], color=ORANGE, marker="s", ms=4, lw=1.2, ls=(0, (3, 1.5)), label="Worst pair")
a.axvline(dep, color=GREY, lw=0.6, ls=":")
a.set_yscale("log")
a.set_ylim(1e-3, 1)
a.set_yticks([1e-3, 1e-2, 1e-1, 1])
a.set_xticks(xs)
a.set_xticklabels(lab)
a.set_xlabel("JPEG quality / median payload (KB)")
a.set_ylabel("1 − IoU with PyTorch box (log)")
a.legend(loc="upper right")
a.grid(axis="x", visible=False)
save(fig, "figures/jpeg_boxes.png")

fig, b = panel()
b.plot(xs, [r["cam_median"] for r in R], color=BLUE, marker="o", ms=4, lw=1.2, label="Median frame")
b.plot(xs, [r["cam_max"] for r in R], color=ORANGE, marker="s", ms=4, lw=1.2, ls=(0, (3, 1.5)), label="Worst frame")
b.axvline(dep, color=GREY, lw=0.6, ls=":")
b.set_yscale("log")
b.set_ylim(5e-3, 1)
b.set_yticks([1e-2, 1e-1, 1])
b.set_xticks(xs)
b.set_xticklabels(lab)
b.set_xlabel("JPEG quality / median payload (KB)")
b.set_ylabel("Max abs. heat-map difference (log)")
b.legend(loc="upper right")
b.grid(axis="x", visible=False)
save(fig, "figures/jpeg_cam.png")
