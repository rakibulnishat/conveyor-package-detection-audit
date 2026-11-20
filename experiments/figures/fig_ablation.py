"""Single-run ablation recorded in the training notebook (test split of the frame-random Roboflow split)."""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/figures")
from figstyle import BLUE, GREY, INK2, ORANGE, panel, save  # noqa: E402

ab = pd.read_csv("experiments/results/ablation/notebook_ablation_test.csv")
dep = json.load(open("experiments/results/deployed_eval/summary.json"))["test"]
names = {"baseline_yolo12s (main run above)": "YOLOv12s, 640 (notebook baseline)", "scale_yolo12n": "YOLOv12n, 640",
         "scale_yolo12m": "YOLOv12m, 640", "aug_no_mosaic": "YOLOv12s, no mosaic", "aug_none": "YOLOv12s, no augmentation",
         "res_416": "YOLOv12s, 416", "res_832": "YOLOv12s, 832"}
ab["label"] = ab.config.map(names)
rows = [("YOLOv12s, 640 (deployed checkpoint)", dep["mAP50_95"], dep["mAP50"])] + \
       list(zip(ab.label, ab.mAP50_95, ab.mAP50))
ys = np.arange(len(rows))[::-1]
for k, xl, lim, fn_, show in ((1, "Test mAP@0.5:0.95", (0.55, 0.80), "ablation_map5095.png", True),
                              (2, "Test mAP@0.5", (0.94, 1.0), "ablation_map50.png", True)):
    fig, ax = panel(3.3, 2.7)
    base = [rows[0][k], rows[1][k]]
    ax.axvspan(min(base), max(base), color="#e8f0fb", zorder=0)
    for y, r in zip(ys, rows):
        c, m = (BLUE, "o") if "640 (" in r[0] else ((ORANGE, "s") if "no augmentation" in r[0] else (GREY, "D"))
        ax.plot([lim[0], r[k]], [y, y], color="#e3e2de", lw=0.8, zorder=1)
        ax.scatter([r[k]], [y], s=20, marker=m, color=c, zorder=3, edgecolor="white", linewidth=0.5)
        ax.text(r[k] + (lim[1] - lim[0]) * 0.02, y, f"{r[k]:.3f}", va="center", fontsize=6.8, color=INK2)
    ax.set_xlim(*lim)
    ax.set_xlabel(xl)
    ax.grid(axis="y", visible=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0].replace(" (", "\n(") for r in rows], fontsize=6.8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    save(fig, "figures/" + fn_)
