"""Stage 2: stratified train/val/test split, written as COCO-format JSON.

COCO is the interchange format of the detection world (CVAT/Label Studio export it,
pycocotools evaluates it). Minimal structure:
    images:      [{"id", "file_name", "width", "height"}]
    annotations: [{"id", "image_id", "category_id", "bbox": [x, y, w, h], "area", "iscrowd"}]
    categories:  [{"id", "name"}]
Note COCO bbox is [x_min, y_min, width, height], not xyxy.

Split per class so every class keeps the same ratio in each split (stratified).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image

P = yaml.safe_load(Path("params.yaml").read_text())
S = P["split"]
IMAGES, SIZE, OUT = Path(P["data"]["images_dir"]), P["data"]["image_size"], Path("data/coco")
classes = list(P["classes"])

by_class = defaultdict(list)
for rec in json.loads(Path("data/autolabels.json").read_text()):
    by_class[rec["class_id"]].append(rec)

rng = random.Random(S["seed"])
splits = {"train": [], "val": [], "test": []}
for cls_id, recs in sorted(by_class.items()):
    rng.shuffle(recs)
    n_val, n_test = max(1, int(len(recs) * S["val"])), max(1, int(len(recs) * S["test"]))
    splits["val"] += recs[:n_val]
    splits["test"] += recs[n_val:n_val + n_test]
    splits["train"] += recs[n_val + n_test:]
    print(f"{classes[cls_id]:12s} train={len(recs) - n_val - n_test} val={n_val} test={n_test}")

OUT.mkdir(parents=True, exist_ok=True)
categories = [{"id": i + 1, "name": name} for i, name in enumerate(classes)]  # COCO ids are 1-based
ann_id = 1
for split, recs in splits.items():
    images, annotations = [], []
    for img_id, rec in enumerate(recs, 1):
        w, h = Image.open(IMAGES / rec["file"]).size
        # ponytail: training reads images as-is, so they must already be letterboxed to SIZE;
        # add a resize-with-boxes step in train.py to use arbitrary-size images
        assert (w, h) == (SIZE, SIZE), f"{rec['file']} is {w}x{h}, expected {SIZE}x{SIZE}"
        images.append({"id": img_id, "file_name": rec["file"], "width": w, "height": h})
        for x1, y1, x2, y2 in rec["boxes"]:
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": rec["class_id"] + 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            ann_id += 1
    (OUT / f"{split}.json").write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
