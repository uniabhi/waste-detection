"""Stage 1: turn the classification dataset into a detection dataset.

The folder name already tells us WHAT is in each image; Grounding DINO (a zero-shot,
text-prompted detector) tells us WHERE. Output: data/autolabels.json, one record per image:
    {"file": "battery/battery_1.jpg", "class_id": 0, "boxes": [[x1, y1, x2, y2], ...], "scores": [...]}
Boxes are absolute pixels in xyxy format.

This is the only PyTorch file in the repo: Grounding DINO has no TensorFlow implementation.
It's a labeling tool that runs once; the model you train and serve is pure TF/Keras.

Images with no confident box are dropped, NOT kept as empty "background" images:
an unlabeled battery photo would teach the model that batteries are background.
"""
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

P = yaml.safe_load(Path("params.yaml").read_text())
A = P["autolabel"]
IMAGES, REPORTS = Path(P["data"]["images_dir"]), Path("reports")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(A["model"])
model = AutoModelForZeroShotObjectDetection.from_pretrained(A["model"]).to(device).eval()

records, stats, rejected = [], {}, []
for cls_id, (cls, prompts) in enumerate(P["classes"].items()):
    text = ". ".join(prompts).lower() + "."  # Grounding DINO wants lowercase "a. b. c."
    files = sorted(f for f in (IMAGES / cls).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = files[: A["limit_per_class"] or None]
    n_labeled = n_boxes = 0
    # ponytail: batch size 1 (~1 h for 12k imgs on a T4, runs once, DVC caches it); batch images to speed up
    for f in tqdm(files, desc=cls):
        img = Image.open(f).convert("RGB")
        w, h = img.size
        inputs = processor(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        r = processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=A["box_threshold"],
            text_threshold=A["text_threshold"], target_sizes=[(h, w)])[0]
        boxes, scores = r["boxes"].cpu(), r["scores"].cpu()
        keep = nms(boxes, scores, A["nms_iou"])[: A["max_boxes"]]  # prompts overlap -> duplicate boxes
        if not len(keep):
            rejected.append(str(f))
            continue
        boxes = boxes[keep].clamp(min=0)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(max=h)
        records.append({"file": f"{cls}/{f.name}", "class_id": cls_id,
                        "boxes": boxes.round(decimals=1).tolist(), "scores": scores[keep].round(decimals=3).tolist()})
        n_labeled, n_boxes = n_labeled + 1, n_boxes + len(keep)
    stats[cls] = {"images": len(files), "labeled": n_labeled, "boxes": n_boxes}

Path("data").mkdir(exist_ok=True)
Path("data/autolabels.json").write_text(json.dumps(records))
REPORTS.mkdir(exist_ok=True)
(REPORTS / "autolabel.json").write_text(json.dumps(stats, indent=2))
(REPORTS / "autolabel_rejected.txt").write_text("\n".join(rejected) + "\n")
print(json.dumps(stats, indent=2))
