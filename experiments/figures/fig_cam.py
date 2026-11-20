"""Eigen-CAM checks on 191 clip frames. cam_curves.png (a): mean deletion and insertion curves;
cam_randomisation.png (b): Spearman correlation of each map with the trained map."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/figures")
from figstyle import AQUA, BLUE, GREY, INK2, ORANGE, panel, save  # noqa: E402

D = json.load(open("experiments/results/eigencam/deletion_insertion.json"))
x = np.array(D["steps"]) * 100
F = D["frames"]
fig, a = panel()
for key, c, m, name in (("cam", BLUE, "o", "Eigen-CAM order"), ("rand", GREY, "s", "Random order"), ("inv", ORANGE, "^", "Reversed order")):
    for mode, ls in (("del", "-"), ("ins", (0, (3, 1.5)))):
        y = np.mean([f[f"{key}_{mode}_curve"] for f in F], 0)
        a.plot(x, y, color=c, ls=ls, marker=m, ms=3.2, lw=1.2, label=f"{name}, {'deletion' if mode == 'del' else 'insertion'}")
a.set_xlabel("Cells greyed out (deletion) or revealed (insertion), %")
a.set_ylabel("Mean top package score")
a.set_ylim(0, 1.32)
a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a.set_xlim(-3, 103)
a.legend(loc="upper center", ncol=2, fontsize=6.3, bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=0.8)
save(fig, "figures/cam_curves.png")

R = pd.read_csv("experiments/results/eigencam/randomisation_spearman.csv")
cols = [("rho_onnx", "Deployed ONNX", BLUE, "o"), ("rho_coco", "COCO-pretrained", AQUA, "s"),
        ("rho_random_s0", "Random, seed 0", GREY, "D"), ("rho_random_s1", "Random, seed 1", GREY, "D"),
        ("rho_random_s2", "Random, seed 2", GREY, "D"), ("rho_sobel", "Sobel edges", ORANGE, "^")]
rng = np.random.default_rng(0)
fig, b = panel()
for i, (k, name, c, m) in enumerate(cols):
    v = R[k].values
    b.scatter(v, i + rng.uniform(-0.17, 0.17, len(v)), s=5, color=c, marker=m, alpha=0.55, linewidth=0, zorder=3)
    b.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color="#0b0b0b", lw=1.1, zorder=4)
b.axvline(0, color=INK2, lw=0.6)
b.set_yticks(range(len(cols)))
b.set_yticklabels([c[1] for c in cols])
b.invert_yaxis()
b.set_xlim(-1.05, 1.05)
b.set_xlabel("Spearman \u03c1 with the trained map")
b.grid(axis="y", visible=False)
save(fig, "figures/cam_randomisation.png")
