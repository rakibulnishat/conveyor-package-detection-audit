"""Server-side latency budget of the deployed ONNX path on the test CPU (measured), and the sampling rate the client can
sustain with two requests in flight (analytical upper bound: rate = n / (server time + network time), n requests in flight).
latency_stages.png (a), latency_rate.png (b)."""
import json
import sys

import numpy as np

sys.path.insert(0, "experiments/figures")
from figstyle import AQUA, BLUE, GREY, INK2, ORANGE, panel, save  # noqa: E402

L = {t: json.load(open(f"experiments/results/latency/latency_t{t}.json")) for t in (1, 2)}
stages = [("decode", "JPEG decode", GREY), ("prep", "Letterbox + layout", AQUA), ("infer", "ONNX inference", BLUE),
          ("post", "Filter + NMS", ORANGE), ("cam", "Eigen-CAM grid", "#6250d6")]
fig, a = panel()
ys = [1, 0]
for y, t in zip(ys, (1, 2)):
    left = 0
    for k, name, c in stages:
        w = L[t]["stage_ms"][k]["median"]
        a.barh(y, w, left=left, height=0.5, color=c, edgecolor="white", linewidth=0.6, label=name if t == 1 else None)
        left += w
    a.text(left + 3, y, f"{L[t]['total_ms']['median']:.0f} ms (p95 {L[t]['total_ms']['p95']:.0f})", va="center",
           fontsize=6.8, color=INK2)
pt = L[1]["pytorch_predict_ms"]["median"]
a.barh(-1, pt, height=0.5, color="#c9c8c3", edgecolor="white")
a.text(pt + 3, -1, f"{pt:.0f} ms", va="center", fontsize=6.8, color=INK2)
a.set_yticks([1, 0, -1])
a.set_yticklabels(["ONNX\n1 thread", "ONNX\n2 threads", "PyTorch\npredict"])
a.set_xlim(0, 260)
a.set_ylim(-1.6, 2.6)
a.set_xlabel("Median time per frame (ms)")
a.legend(loc="upper center", ncol=2, fontsize=6.4, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
a.grid(axis="y", visible=False)
save(fig, "figures/latency_stages.png")

net = np.linspace(0, 400, 401)
fig, b = panel()
T2 = L[2]["total_ms"]["median"]
for k, c, ls, lab in ((2, BLUE, "-", "Video file, 2 requests in flight"), (1, ORANGE, (0, (3, 1.5)), "Live camera, 1 request in flight")):
    b.plot(net, 1000 * k / (T2 + net), color=c, ls=ls, lw=1.4, label=lab)
for r in (4, 6, 10, 15):
    b.axhline(r, color="#c9c8c3", lw=0.7, zorder=0)
b.set_xlim(0, 400)
b.set_ylim(0, 25)
b.set_yticks([0, 4, 6, 10, 15, 20, 25])
b.grid(axis="y", visible=False)
b.set_xlabel("Network round trip per request (ms)")
b.set_ylabel("Upper bound on rate (frames s$^{-1}$)")
b.legend(loc="upper center", bbox_to_anchor=(0.6, 1.0))
save(fig, "figures/latency_rate.png")
