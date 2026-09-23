"""Inference around the exported SavedModel: letterbox -> model -> map boxes back to the original image.

The SavedModel already contains normalization, anchor decoding and NMS (see src/train.py),
so serving only needs TensorFlow, not Keras/KerasHub.
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def letterbox(img: Image.Image, size: int):
    """Resize keeping aspect ratio, pad to size x size with gray 114 (exactly how the dataset images were made)."""
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas, r, (px, py)


class Detector:
    def __init__(self, path: str):
        self.serve = tf.saved_model.load(path).serve
        meta = json.loads((Path(path) / "meta.json").read_text())  # written by train.py
        self.names, self.size = meta["classes"], meta["image_size"]

    def __call__(self, img: Image.Image, conf=0.25):
        img = img.convert("RGB")
        x, r, (px, py) = letterbox(img, self.size)
        out = {k: v.numpy()[0] for k, v in self.serve(tf.constant(np.asarray(x, np.float32)[None])).items()}
        n = int(out["num_detections"])
        keep = out["confidence"][:n] >= conf
        boxes = (out["boxes"][:n][keep] - [px, py, px, py]) / r  # undo letterbox -> original pixels
        boxes = boxes.clip(0, [img.width, img.height] * 2)
        return [{"class": self.names[int(c)], "confidence": round(float(s), 4), "box": [round(float(v), 1) for v in b]}
                for b, s, c in zip(boxes, out["confidence"][:n][keep], out["labels"][:n][keep])]
