"""Deployed checkpoint on the Roboflow validation and test splits (CPU re-evaluation, Ultralytics 8.4.160)."""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/figures")
from figstyle import AQUA, INK, INK2, ORANGE, panel, save  # noqa: E402

S = json.load(open("experiments/results/deployed_eval/summary.json"))
fig, a = panel()
for s, c, m, lab in (("test", ORANGE, "o", "Test"), ("val", AQUA, "s", "Validation")):
    d = pd.read_csv(f"experiments/results/deployed_eval/{s}_curve_Precision_RecallB.csv")
    a.plot(d.x, d.iloc[:, 1], color=c, lw=1.3, label=f"{lab}, AP@0.5 {S[s]['mAP50']:.3f}")
    cm = np.array(S[s]["confusion_matrix"])
    tp, fp, fn = cm[0, 0], cm[0, 1], cm[1, 0]
    a.scatter([tp / (tp + fn)], [tp / (tp + fp)], s=26, marker=m, color=c, edgecolor="white", linewidth=0.6, zorder=4)
a.scatter([], [], s=26, marker="o", color="#9a9994", label="Confidence 0.25")
a.set_xlim(0, 1.02)
a.set_ylim(0, 1.05)
a.set_xlabel("Recall")
a.set_ylabel("Precision")
a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02))
save(fig, "figures/accuracy_pr.png")

for s, fn_ in (("test", "accuracy_cm_test.png"), ("val", "accuracy_cm_val.png")):
    fig, ax = panel(2.3, 2.3)
    cm = np.array(S[s]["confusion_matrix"])  # rows predicted, columns true (Ultralytics convention)
    shade = np.array([[1, 0.35], [0.35, 0.0]])
    ax.imshow(shade, cmap="Blues", vmin=0, vmax=1.6)
    names = [["TP", "FP"], ["FN", "–"]]
    for i in range(2):
        for j in range(2):
            v = "" if (i == 1 and j == 1) else f"{int(cm[i, j])}"
            ax.text(j, i - 0.12, v, ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if (i, j) == (0, 0) else INK)
            ax.text(j, i + 0.22, names[i][j], ha="center", va="center", fontsize=6.5,
                    color="white" if (i, j) == (0, 0) else INK2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["package", "background"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["package", "background"], rotation=90, va="center")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "figures/" + fn_)
