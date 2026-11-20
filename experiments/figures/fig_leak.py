"""Leakage experiment on the shared 16-frame test block. leakage_metrics.png (a), leakage_packages.png (b)."""
import json
import sys

import numpy as np

sys.path.insert(0, "experiments/figures")
from figstyle import BLUE, GREY, INK2, ORANGE, panel, save  # noqa: E402

S = json.load(open("experiments/results/leakage/leakage_summary.json"))
C = (("leaky", "Leaky split", ORANGE, "o"), ("blocked", "Time-blocked split", BLUE, "s"))
metrics = (("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95"), ("recall_025", "Recall at 0.25"))
fig, a = panel()
xs = np.arange(len(metrics))
for k, (c, name, col, m) in enumerate(C):
    off = (k - 0.5) * 0.28
    for i, (key, _) in enumerate(metrics):
        v = S[c][key]
        a.scatter([i + off], [v], s=30, color=col, marker=m, zorder=3, edgecolor="white", linewidth=0.5,
                  label=name if i == 0 else None)
        top = v
        if key == "recall_025":
            lo, hi = S[c]["recall_ci"]
            a.plot([i + off] * 2, [lo, hi], color=col, lw=1.1, zorder=2)
            top = hi
        a.text(i + off, top + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=6.6, color=INK2)
a.set_xticks(xs)
a.set_xticklabels([m[1] for m in metrics])
a.set_xlim(-0.6, len(metrics) - 0.4)
a.set_ylim(0, 1.18)
a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a.set_ylabel("Test-block value")
a.legend(loc="lower left")
a.grid(axis="x", visible=False)
save(fig, "figures/leakage_metrics.png")

pk = sorted(int(p) for p in S["leaky"]["per_package"] if int(p) >= 0)
fig, b = panel()
for k, (c, name, col, m) in enumerate(C):
    off = (k - 0.5) * 0.28
    for i, p in enumerate(pk):
        d = S[c]["per_package"][str(p)]
        b.scatter([i + off], [d["recall"]], s=30, color=col, marker=m, zorder=3, edgecolor="white", linewidth=0.5,
                  label=name if i == 0 else None)
        b.plot([i + off] * 2, d["ci"], color=col, lw=1.1, zorder=2)
        b.text(i + off, d["ci"][1] + 0.025, f"{d['k']}/{d['n']}", ha="center", va="bottom", fontsize=6.6, color=INK2)
b.set_xticks(range(len(pk)))
b.set_xticklabels([f"Package {p}" for p in pk])
b.set_xlim(-0.6, len(pk) - 0.4)
b.set_ylim(0, 1.18)
b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
b.set_ylabel("Recall at 0.25 (95% Wilson CI)")
b.legend(loc="lower left")
b.grid(axis="x", visible=False)
save(fig, "figures/leakage_packages.png")
