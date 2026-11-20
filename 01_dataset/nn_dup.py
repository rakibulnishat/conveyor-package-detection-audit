import sys, glob, os, csv, numpy as np
from PIL import Image
DS, OUT = sys.argv[1], sys.argv[2]
def load(p, s=128):
    return np.asarray(Image.open(p).convert('L').resize((s, s), Image.BILINEAR), np.float32) / 255.
def boxes(p):
    lab = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    return [list(map(float, l.split()[1:])) for l in open(lab).read().split('\n') if l.strip()]
sp = {s: sorted(glob.glob(f'{DS}/{s}/images/*')) for s in ('train', 'valid', 'test')}
X = {s: np.stack([load(p) for p in sp[s]]) for s in sp}
def ncc(a, B):
    a = (a - a.mean()) / (a.std() + 1e-6); Bn = (B - B.mean((1, 2), keepdims=True)) / (B.std((1, 2), keepdims=True) + 1e-6)
    return (Bn * a).mean((1, 2))
rows = []
for s in ('test', 'valid'):
    for i, p in enumerate(sp[s]):
        mae = np.abs(X['train'] - X[s][i]).mean((1, 2)); j = int(mae.argmin())
        c = ncc(X[s][i], X['train'])[j]
        rows.append(dict(split=s, file=os.path.basename(p), nn_train=os.path.basename(sp['train'][j]), mae=round(float(mae[j]), 4),
                         ncc=round(float(c), 4), n_box=len(boxes(p)), n_box_nn=len(boxes(sp['train'][j]))))
# reference: within-train nearest neighbour (excluding self)
wt = []
for i in range(len(sp['train'])):
    mae = np.abs(X['train'] - X['train'][i]).mean((1, 2)); mae[i] = 9; wt.append(float(mae.min()))
with open(f'{OUT}/nearest_train_neighbour.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for s in ('test', 'valid'):
    m = np.array([r['mae'] for r in rows if r['split'] == s]); c = np.array([r['ncc'] for r in rows if r['split'] == s])
    print(s, 'nearest-train MAE: min %.4f median %.4f max %.4f | NCC median %.3f | n MAE<0.02: %d/%d' % (m.min(), np.median(m), m.max(), np.median(c), (m < 0.02).sum(), len(m)))
print('train->train nearest MAE: min %.4f median %.4f max %.4f' % (min(wt), np.median(wt), max(wt)))
for r in sorted([r for r in rows if r['split'] == 'test'], key=lambda r: r['mae'])[:19]: print(r)
