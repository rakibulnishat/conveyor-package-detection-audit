"""Build the two training splits of the leakage experiment from the Roboflow export.

Run from the repository root:  python experiments/07_leakage/make_splits.py
Reads   data/roboflow/{train,valid,test}/{images,labels} and experiments/07_leakage/preregistration.json
Writes  experiments/07_leakage/work/{leaky,blocked}/{train,valid,test}/{images,labels} and data.yaml

The frame lists come from preregistration.json. The 16 frames dropped from the leaky training set were drawn with
random.Random(0).sample(blocked_train, 16); the check below confirms that the stored list equals that draw.
"""
import glob
import json
import os
import random
import re
import shutil

SRC = 'data/roboflow'
WORK = 'experiments/07_leakage/work'
P = json.load(open('experiments/07_leakage/preregistration.json'))

assert sorted(random.Random(0).sample(sorted(P['blocked_train']), 16)) == sorted(P['dropped_from_leaky'])

files = {}
for img in glob.glob(f'{SRC}/*/images/*'):
    k = int(re.search(r'frame_(\d+)', os.path.basename(img)).group(1))
    lab = img.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    files[k] = (img, lab)
assert len(files) == 191

splits = {'leaky': {'train': P['leaky_train'], 'valid': P['val_frames'], 'test': P['test_frames']},
          'blocked': {'train': P['blocked_train'], 'valid': P['val_frames'], 'test': P['test_frames']}}
for cond, parts in splits.items():
    root = os.path.abspath(f'{WORK}/{cond}')
    shutil.rmtree(root, ignore_errors=True)
    for part, frames in parts.items():
        for sub in ('images', 'labels'):
            os.makedirs(f'{root}/{part}/{sub}')
        for k in frames:
            img, lab = files[k]
            shutil.copy2(img, f'{root}/{part}/images/')
            shutil.copy2(lab, f'{root}/{part}/labels/')
    with open(f'{root}/data.yaml', 'w') as f:
        f.write(f"path: {root}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['object']\n")
    print(cond, {p: len(v) for p, v in parts.items()})
