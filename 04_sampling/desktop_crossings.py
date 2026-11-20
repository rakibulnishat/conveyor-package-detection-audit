"""Crossing events of the desktop path at the full frame rate.

Run from the repository root:  python experiments/04_sampling/desktop_crossings.py [out.json]
Reads experiments/results/reference/pt_track.json (Ultralytics model.track on every frame, as the desktop app calls it)
and applies the desktop window rule |cx - 418| < 18 once per track id. Event = [frame index, time in s].
"""
import json
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else 'experiments/results/sampling/desktop_crossings.json'
LINE_X, TOL = 418, 18
d = json.load(open('experiments/results/reference/pt_track.json'))
events = {}
for f in d['frames']:
    for b in f['boxes']:
        if len(b) > 6:
            tid = int(b[6]); cx = (b[0] + b[2]) / 2
            if abs(cx - LINE_X) < TOL and str(tid) not in events:
                events[str(tid)] = [f['i'], round(f['t'], 3)]
ids = sorted({int(b[6]) for f in d['frames'] for b in f['boxes'] if len(b) > 6})
json.dump({'line_x': LINE_X, 'tol': TOL, 'ids': ids, 'crossings': events}, open(OUT, 'w'))
print(events)
