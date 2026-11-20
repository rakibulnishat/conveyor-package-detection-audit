"""Crossing events from the deployed browser tracker (pipeline.js in Node) under the web sampler
(seek to phase + k/rate; UI rates 4, 6, 10, 15 frames/s, and every frame), all start phases.
crossings_count.png (a), crossings_delay.png (b)."""
import json
import sys

import numpy as np

sys.path.insert(0, "experiments/figures")
from figstyle import BLUE, INK2, ORANGE, panel, save  # noqa: E402

R = json.load(open("experiments/results/sampling/web_rates_jpeg85.json"))
rates = [29.97, 15, 10, 6, 4]
xs = np.arange(len(rates))
lab = ["29.97", "15", "10", "6", "4"]
rng = np.random.default_rng(1)
fig, a = panel()
for k, (rule, c, m, name) in enumerate((("window_only", ORANGE, "s", "Window rule only (desktop)"),
                                         ("window_or_signchange", BLUE, "o", "Window or sign change (web)"))):
    for i, r in enumerate(rates):
        v = np.array([x["events"] for x in R if x["rule"] == rule and x["rate"] == r])
        off = (k - 0.5) * 0.3
        a.scatter(i + off + rng.uniform(-0.07, 0.07, len(v)), v + rng.uniform(-0.05, 0.05, len(v)), s=12, marker=m, color=c,
                  edgecolor="white", linewidth=0.4, zorder=3, label=name if i == 0 else None)
a.axhline(3, color=INK2, lw=0.6, ls=":")
a.set_xticks(xs)
a.set_xticklabels(lab)
a.set_xlabel("Sampling rate (frames s$^{-1}$)")
a.set_ylabel("Packages counted (of 3)")
a.set_yticks([0, 1, 2, 3])
a.set_ylim(-0.3, 4.4)
a.legend(loc="upper right")
a.grid(axis="x", visible=False)
save(fig, "figures/crossings_count.png")

ref = {i: t for i, t, _ in next(x for x in R if x["rate"] == 29.97 and x["rule"] == "window_or_signchange")["times"]}
fig, b = panel()
for i, r in enumerate(rates):
    d = np.array([(t - ref[pid]) * 1000 for x in R if x["rule"] == "window_or_signchange" and x["rate"] == r
                  for pid, t, _ in x["times"]])
    b.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=12, color=BLUE, edgecolor="white", linewidth=0.4, zorder=3)
    if d.max() > 0:
        b.plot([i - 0.25, i + 0.25], [d.max()] * 2, color=INK2, lw=0.8)
        b.text(i, d.max() + 10, f"max {d.max():.0f}", ha="center", va="bottom", fontsize=6.8, color=INK2)
b.axhline(0, color=INK2, lw=0.6)
b.set_xticks(xs)
b.set_xticklabels(lab)
b.set_xlim(-0.5, len(rates) - 0.2)
b.set_xlabel("Sampling rate (frames s$^{-1}$)")
b.set_ylabel("Delay vs 29.97 frames s$^{-1}$ (ms)")
b.set_ylim(-30, 300)
b.grid(axis="x", visible=False)
save(fig, "figures/crossings_delay.png")
