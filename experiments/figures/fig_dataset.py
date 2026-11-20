"""Dataset provenance: every image is a frame of co.mp4; splits on the frame timeline and nearest-neighbour distances."""
import json
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, "experiments/figures")
from figstyle import AQUA, BLUE, GREY, GREY_L, INK2, ORANGE, panel, save  # noqa: E402

gt = pd.read_csv("experiments/results/dataset/gt_boxes_by_frame.csv")
split = gt.groupby("frame_no")["split"].first()
nbox = gt.groupby("frame_no").size()
pr = json.load(open("experiments/07_leakage/preregistration.json"))
N = 191
COL = {"train": BLUE, "valid": AQUA, "test": ORANGE, "unused": "#ffffff"}


def design(train):
    lab = {}
    for k in range(1, N + 1):
        lab[k] = "unused"
    for k in train:
        lab[k] = "train"
    for k in pr["val_frames"]:
        lab[k] = "valid"
    for k in pr["test_frames"]:
        lab[k] = "test"
    return lab


rows = [("Roboflow split\n(deployed model)", {k: split[k] for k in range(1, N + 1)}),
        ("Leaky design", design(pr["leaky_train"])),
        ("Time-blocked design", design(pr["blocked_train"]))]

fig, a = panel(6.8, 2.15)
for r, (name, lab) in enumerate(rows):
    y = len(rows) - 1 - r
    for k in range(1, N + 1):
        c = COL[lab[k]]
        a.add_patch(plt.Rectangle((k - 0.5, y + 0.12), 1.0, 0.76, facecolor=c, edgecolor=GREY_L if c == "#ffffff" else "white",
                                  linewidth=0.15))
# labelled boxes per frame as a thin bar track on top
top = len(rows) + 0.15
for k in range(1, N + 1):
    h = nbox.get(k, 0) * 0.18
    a.add_patch(plt.Rectangle((k - 0.4, top), 0.8, h, facecolor=GREY, edgecolor="none"))
a.set_xlim(0, N + 1)
a.set_ylim(-0.1, top + 4 * 0.18 + 0.1)
a.set_yticks([len(rows) - 1 - r + 0.5 for r in range(len(rows))] + [top + 0.36])
a.set_yticklabels([n for n, _ in rows] + ["labelled boxes\n(1–4 per frame)"])
a.set_xticks([1, 30, 60, 90, 120, 150, 191])
a.set_xlabel("Frame index in the 6.4 s source clip (29.97 frames s$^{-1}$)")
a.grid(False)
a.spines["left"].set_visible(False)
a.tick_params(axis="y", length=0)
a.legend(handles=[Patch(facecolor=BLUE, label="train"), Patch(facecolor=AQUA, label="validation"),
                  Patch(facecolor=ORANGE, label="test"), Patch(facecolor="white", edgecolor=GREY_L, label="not used")],
         ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
save(fig, "figures/dataset_timeline.png")

# (b) nearest training-frame distance
nn = pd.read_csv("experiments/results/dataset/nearest_train_neighbour.csv")
lk = pd.read_csv("experiments/results/leakage/test_nn_by_condition.csv")
groups = [("Roboflow\nvalidation", nn[nn.split == "valid"].mae.values, AQUA),
          ("Roboflow\ntest", nn[nn.split == "test"].mae.values, ORANGE),
          ("Leaky\ntest block", lk[lk.condition == "leaky"].mae.values, ORANGE),
          ("Blocked\ntest block", lk[lk.condition == "blocked"].mae.values, GREY)]
fig, b = panel()
rng = np.random.default_rng(0)
for i, (name, v, c) in enumerate(groups):
    x = i + rng.uniform(-0.18, 0.18, len(v))
    b.scatter(x, v, s=9, color=c, edgecolor="white", linewidth=0.4, zorder=3)
    b.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="#0b0b0b", lw=1.0, zorder=4)
b.set_xticks(range(len(groups)))
b.set_xticklabels([g[0] for g in groups])
b.set_ylabel("MAE to nearest training frame")
b.set_ylim(0, 0.065)
b.set_xlim(-0.6, len(groups) - 0.4)
b.grid(axis="x", visible=False)
save(fig, "figures/dataset_nn_distance.png")

# (c) frame offset of that neighbour
fig, c_ = panel()
off_rf = np.abs(nn.nn_train.map(lambda s: int(re.search(r"frame_(\d+)", s).group(1))) -
                nn.file.map(lambda s: int(re.search(r"frame_(\d+)", s).group(1)))).values
vals = [("Roboflow\nval + test", off_rf, ORANGE), ("Leaky\ntest block", lk[lk.condition == "leaky"].offset.values, ORANGE),
        ("Blocked\ntest block", lk[lk.condition == "blocked"].offset.values, GREY)]
for i, (name, v, col) in enumerate(vals):
    y = i + rng.uniform(-0.15, 0.15, len(v))
    c_.scatter(v, y, s=9, color=col, edgecolor="white", linewidth=0.4, zorder=3)
    c_.text(np.max(v) * 1.6, i, f"median {np.median(v):g}\n(n = {len(v)})", va="center", fontsize=6.5, color=INK2)
c_.set_xscale("log")
c_.set_xlim(0.6, 3000)
c_.set_ylim(-0.6, len(vals) - 0.4)
c_.set_yticks(range(len(vals)))
c_.set_yticklabels([v[0] for v in vals])
c_.invert_yaxis()
c_.set_xlabel("Frames to the closest training frame (log)")
c_.grid(axis="y", visible=False)
save(fig, "figures/dataset_nn_offset.png")
