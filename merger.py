import os
import sys
import json

from collections import defaultdict

"""Merge the metadata.json of different features"""

dataset_dir = sys.argv[1]
sub_dirs = [i for i in os.listdir(sys.argv[1]) if 'json' not in i]



metas = []
merged = {}

for sub_dir in sub_dirs:
    metas.append(json.load(open(os.path.join(dataset_dir, sub_dir, 'metadata.json'))))

feature_names = set()
for key in metas[0].keys():
    if key == 'feature_name':
        continue
    merged[key] = defaultdict(dict)
    for subdir, meta in zip(sub_dirs, metas):
        for value in meta[key]:
            merged[key][value['audio_path']]['audio_path'] = value['audio_path']
            merged[key][value['audio_path']]['feature_name'] = os.path.join(subdir, value['feature_path'])
            feature_names.add(meta['feature_name'])

# Remove things that don't have both features
before = 0
after = 0
for key in merged:
    before += len(merged[key])
    merged[key] = [row for row in merged[key].values() if feature_names <= set(row.keys())]
    after += len(merged[key])
print(f"Merging, {before} before and {after} after removing things without all features")
json.dump(merged, open(os.path.join(dataset_dir, 'metadata.json'), 'w'), indent=2)
