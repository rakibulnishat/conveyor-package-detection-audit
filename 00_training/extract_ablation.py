"""Copy the single-run ablation table out of the training notebook's output into a CSV.

Run from the repository root:  python experiments/00_training/extract_ablation.py [out.csv]
The table is the pandas display printed by the last cell of train_deployed_model.ipynb (test split, one run per row).
"""
import json
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else 'experiments/results/ablation/notebook_ablation_test.csv'
nb = json.load(open('experiments/00_training/train_deployed_model.ipynb'))
text = None
for c in nb['cells']:
    for o in c.get('outputs', []):
        t = ''.join(o.get('data', {}).get('text/plain', []))
        if 'config_name' in t:
            text = t
lines = text.splitlines()
metrics, times = {}, {}
for ln in lines:
    m = re.match(r'^(\d+)\s+(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\\?$', ln)
    if m:
        metrics[int(m.group(1))] = (m.group(2), *m.group(3, 4, 5, 6))
    m = re.match(r'^(\d+)\s+(NaN|[\d.]+)\s+\S', ln)
    if m and int(m.group(1)) in metrics:
        times[int(m.group(1))] = m.group(2)
fmt = lambda v: repr(float(v)) if v != 'NaN' else 'NaN'
with open(OUT, 'w') as f:
    f.write('config,precision,recall,mAP50,mAP50_95,train_time_s\n')
    for i in sorted(metrics):
        name, p, r, a, b = metrics[i]
        f.write(','.join([name, fmt(p), fmt(r), fmt(a), fmt(b), fmt(times[i])]) + '\n')
print(open(OUT).read())
