"""Training log of the deployed checkpoint (100 epochs), read from the results stored in best.pt.
Panels: training_losses.png (a), training_metrics.png (b)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/figures")
from figstyle import AQUA, BLUE, GREY, INK2, ORANGE, panel, save  # noqa: E402

d = pd.read_csv("experiments/results/deployed_eval/train_results.csv")
e = d["epoch"].values
OUT = "figures/"

fig, a1 = panel()
for col, c, m, lab in (("box_loss", BLUE, "o", "box"), ("cls_loss", ORANGE, "s", "cls"), ("dfl_loss", AQUA, "^", "dfl")):
    a1.plot(e, d[f"train/{col}"], color=c, lw=1.2, label=f"train {lab}")
    a1.plot(e, d[f"val/{col}"].values, color=c, lw=0.9, ls=(0, (3, 1.5)), marker=m, ms=2.2, markevery=8, label=f"val {lab}")
a1.set_yscale("log")
a1.set_xlabel("Epoch")
a1.set_ylabel("Loss (log scale)")
a1.set_xlim(0, 101)
nan = np.isnan(d["val/box_loss"].values)
a1.axvspan(e[nan].min() - 0.5, e[nan].max() + 0.5, color="#efeeea", zorder=0)
a1.text(14.5, 0.13, "grey band: validation losses\nnot finite (epochs 3–12)", fontsize=6.8, color=INK2, va="bottom")
a1.set_ylim(0.1, 3000)
a1.set_yticks([0.1, 1, 10, 100, 1000])
a1.legend(ncol=3, loc="upper right", handlelength=2.0, columnspacing=0.8)
save(fig, OUT + "training_losses.png")

fig, a2 = panel()
for col, c, m, lab in (("precision(B)", BLUE, "o", "Precision"), ("recall(B)", ORANGE, "s", "Recall"),
                       ("mAP50(B)", AQUA, "^", "mAP@0.5"), ("mAP50-95(B)", GREY, "D", "mAP@0.5:0.95")):
    a2.plot(e, d[f"metrics/{col}"], color=c, lw=1.1, marker=m, ms=2.2, markevery=10, label=lab)
best = int(d["metrics/mAP50-95(B)"].idxmax())
a2.axvline(e[best], color=INK2, lw=0.6, ls=":")
a2.text(e[best] - 1.5, 0.03, f"best epoch {e[best]}", ha="right", fontsize=6.8, color=INK2)
a2.set_xlabel("Epoch")
a2.set_ylabel("Validation metric")
a2.set_ylim(0, 1.0)
a2.set_xlim(0, 101)
a2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0)
save(fig, OUT + "training_metrics.png")
