"""Desktop PyTorch reference vs deployed ONNX path on all 191 clip frames: raw frames and JPEG q85 frames."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/figures")
from figstyle import BLUE, INK2, ORANGE, panel, save  # noqa: E402

bp = pd.read_csv("experiments/results/parity/box_pairs.csv")
cam = pd.read_csv("experiments/results/parity/cam_unrounded_per_frame.csv")
fig, a = panel()
for mode, c, m, lab in (("raw", BLUE, "o", "Raw frames"), ("jpeg85", ORANGE, "s", "JPEG q = 85 (deployed)")):
    d = bp[bp["mode"] == mode]
    a.scatter(d.frame + 1, 1 - d.iou, s=5, marker=m, color=c, linewidth=0, alpha=0.85, label=f"{lab} ({len(d)} pairs)")
a.set_yscale("log")
a.set_ylim(1e-5, 3)
a.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
a.set_xlim(-2, 195)
a.set_xlabel("Frame index")
a.set_xticks([1, 50, 100, 150, 191])
a.set_ylabel("1 − IoU with PyTorch box (log)")
a.legend(loc="upper left", markerscale=2.2, ncol=1)
save(fig, "figures/parity_boxes.png")
fig, b = panel()

b.plot(cam.frame + 1, cam.raw, color=BLUE, lw=1.0, label="Raw frames")
b.plot(cam.frame + 1, cam.jpeg85, color=ORANGE, lw=1.0, ls=(0, (3, 1.2)), label="JPEG q = 85 (deployed)")
b.set_yscale("log")
b.set_ylim(1e-8, 100)
b.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
b.set_xlim(-2, 195)
b.set_xlabel("Frame index")
b.set_xticks([1, 50, 100, 150, 191])
b.set_ylabel("Max abs. heat-map difference (log)")
b.legend(loc="upper left", ncol=1)
save(fig, "figures/parity_cam.png")
