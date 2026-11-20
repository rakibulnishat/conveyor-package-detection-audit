import sys, glob, os, json, hashlib, csv
import numpy as np
from PIL import Image
DS, OUT = sys.argv[1], sys.argv[2]
rows, hashes, md5 = [], {}, {}
def dhash(p, s=8):
    a = np.asarray(Image.open(p).convert('L').resize((s + 1, s), Image.BILINEAR), dtype=np.int16)
    return int(''.join('1' if v else '0' for v in (a[:, 1:] > a[:, :-1]).flatten()), 2)
for split in ('train', 'valid', 'test'):
    for p in sorted(glob.glob(f'{DS}/{split}/images/*')):
        lab = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        L = [l.split() for l in open(lab).read().split('\n') if l.strip()]
        w, h = Image.open(p).size
        cls = sorted({l[0] for l in L}); bad = [l for l in L if len(l) != 5]
        areas = [float(l[3]) * float(l[4]) for l in L if len(l) == 5]
        hashes[p] = dhash(p); md5[p] = hashlib.md5(open(p, 'rb').read()).hexdigest()
        rows.append(dict(split=split, file=os.path.basename(p), w=w, h=h, n_inst=len(L), classes=';'.join(cls),
                         n_badrows=len(bad), min_area=min(areas) if areas else '', max_area=max(areas) if areas else ''))
with open(f'{OUT}/dataset_audit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summ = {}
for s in ('train', 'valid', 'test'):
    R = [r for r in rows if r['split'] == s]
    summ[s] = dict(images=len(R), instances=sum(r['n_inst'] for r in R), empty=sum(r['n_inst'] == 0 for r in R),
                   max_inst=max(r['n_inst'] for r in R), sizes=sorted({(r['w'], r['h']) for r in R}),
                   bad_rows=sum(r['n_badrows'] for r in R), classes=sorted({c for r in R for c in r['classes'].split(';') if c}))
keys = list(hashes); pairs = []; exact = []
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        si, sj = keys[i].split('/')[-3], keys[j].split('/')[-3]
        d = bin(hashes[keys[i]] ^ hashes[keys[j]]).count('1')
        if md5[keys[i]] == md5[keys[j]]: exact.append((si, sj, os.path.basename(keys[i]), os.path.basename(keys[j])))
        if d <= 4: pairs.append(dict(split_a=si, a=os.path.basename(keys[i]), split_b=sj, b=os.path.basename(keys[j]), hamming=d))
with open(f'{OUT}/near_duplicates_dhash_le4.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['split_a', 'a', 'split_b', 'b', 'hamming']); w.writeheader(); w.writerows(pairs)
summ['exact_duplicate_pairs'] = len(exact)
summ['near_dup_pairs_total'] = len(pairs)
summ['near_dup_pairs_cross_split'] = sum(p['split_a'] != p['split_b'] for p in pairs)
summ['near_dup_by_splitpair'] = {}
for p in pairs:
    k = '-'.join(sorted([p['split_a'], p['split_b']])); summ['near_dup_by_splitpair'][k] = summ['near_dup_by_splitpair'].get(k, 0) + 1
# roboflow source names (files carry original stem before .rf.)
stems = {}
for r in rows:
    st = r['file'].split('.rf.')[0]; stems.setdefault(st, set()).add(r['split'])
summ['source_stems'] = len(stems); summ['stems_in_multiple_splits'] = sum(len(v) > 1 for v in stems.values())
json.dump(summ, open(f'{OUT}/dataset_summary.json', 'w'), indent=1, default=list)
print(json.dumps(summ, indent=1, default=list))
