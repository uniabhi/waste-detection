"""Stage 3: fine-tune a COCO-pretrained RetinaNet (KerasHub) on our boxes, score it on the
untouched test split with COCO mAP, export a TF SavedModel for serving.

RetinaNet = ResNet50 backbone -> FPN (features at several scales, for small + large objects)
-> two heads run on every anchor box: a classifier trained with focal loss (down-weights the
thousands of easy background anchors) and a box regressor (offsets from anchor to object).
"""
import json
from collections import defaultdict
from pathlib import Path

import keras
import keras_hub
import mlflow
import tensorflow as tf
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

P = yaml.safe_load(Path("params.yaml").read_text())
T, SIZE, MAX_BOXES = P["train"], P["data"]["image_size"], P["autolabel"]["max_boxes"]
IMAGES, COCO_DIR, MODELS, REPORTS = Path(P["data"]["images_dir"]), Path("data/coco"), Path("models"), Path("reports")
CLASSES = list(P["classes"])
AUTOTUNE = tf.data.AUTOTUNE


def load(path, boxes, labels):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.ensure_shape(tf.cast(img, tf.float32), [SIZE, SIZE, 3])  # 0-255; the preprocessor normalizes
    return img, {"boxes": boxes, "labels": labels}


def random_flip(img, y):
    """Augmentation must move the boxes with the pixels: a horizontal flip maps x -> SIZE - x."""
    b = y["boxes"]
    flipped = tf.image.flip_left_right(img), {"boxes": tf.stack([SIZE - b[:, 2], b[:, 1], SIZE - b[:, 0], b[:, 3]], 1),
                                              "labels": y["labels"]}
    return tf.cond(tf.random.uniform([]) < 0.5, lambda: flipped, lambda: (img, y))


def dataset(split, train=False):
    """COCO json -> tf.data of (image, {"boxes": xyxy pixels, "labels": 0-based}), batched."""
    coco = json.loads((COCO_DIR / f"{split}.json").read_text())
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    paths = [str(IMAGES / im["file_name"]) for im in coco["images"]]
    boxes = [[[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns[im["id"]])] for im in coco["images"]]
    labels = [[a["category_id"] - 1 for a in anns[im["id"]]] for im in coco["images"]]
    # ragged: every image has a different number of boxes
    ds = tf.data.Dataset.from_tensor_slices((paths, tf.ragged.constant(boxes, ragged_rank=1, dtype=tf.float32),
                                             tf.ragged.constant(labels, dtype=tf.int32)))
    if train:
        ds = ds.shuffle(len(paths))
    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if train:
        ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
    # pad each image to MAX_BOXES boxes (label -1 = "no box") so a batch is one dense tensor
    ds = ds.padded_batch(T["batch_size"], padded_shapes=([SIZE, SIZE, 3], {"boxes": [MAX_BOXES, 4], "labels": [MAX_BOXES]}),
                         padding_values=(0.0, {"boxes": -1.0, "labels": -1}))
    return ds.prefetch(AUTOTUNE)


strategy = tf.distribute.MirroredStrategy()  # one model copy per GPU, gradients averaged; 1 device on CPU
print("devices:", strategy.num_replicas_in_sync)
with strategy.scope():
    preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor.from_preset(T["preset"])  # ImageNet normalization
    preprocessor.image_size = (SIZE, SIZE)  # preset would upscale to its 800x800 COCO size: ~4x the compute
    model = keras_hub.models.RetinaNetObjectDetector(
        backbone=keras_hub.models.Backbone.from_preset(T["preset"]),  # pretrained ResNet50 + FPN
        num_classes=len(CLASSES),  # heads are new: COCO's 80 classes -> our 10
        bounding_box_format="xyxy",
        preprocessor=preprocessor,
        # low threshold so mAP sees the whole precision/recall curve; the API filters with its own conf
        prediction_decoder=keras_hub.layers.NonMaxSuppression(
            bounding_box_format="xyxy", from_logits=True, confidence_threshold=0.05, iou_threshold=0.5, max_detections=100),
    )
    model.compile(optimizer=keras.optimizers.Adam(T["learning_rate"]),
                  box_loss=keras.losses.MeanAbsoluteError(reduction="sum"))

MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
mlflow.set_experiment("waste-detection")  # MLFLOW_TRACKING_URI env var -> remote server, else local
with mlflow.start_run():
    mlflow.log_params({**T, "image_size": SIZE, "max_boxes": MAX_BOXES, "gpus": strategy.num_replicas_in_sync})
    model.fit(dataset("train", train=True), validation_data=dataset("val"), epochs=T["epochs"], callbacks=[
        keras.callbacks.ModelCheckpoint(str(MODELS / "detector.keras"), save_best_only=True),  # survives a Kaggle timeout
        keras.callbacks.EarlyStopping(patience=T["patience"], restore_best_weights=True),
        keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: mlflow.log_metrics(logs, step=epoch)),
    ])

    # --- COCO evaluation on the test split: mAP@[.5:.95], mAP@.5, per-class AP ---
    gt = COCO(str(COCO_DIR / "test.json"))
    images = dataset("test").map(lambda img, y: img)
    pred = model.predict(images)  # preprocess -> network -> decode anchors -> NMS
    dets = []
    for img_id, boxes, scores, labels, n in zip(gt.getImgIds(), pred["boxes"], pred["confidence"],
                                                pred["labels"], pred["num_detections"]):
        for (x1, y1, x2, y2), s, c in zip(boxes[:n].tolist(), scores[:n].tolist(), labels[:n].tolist()):
            dets.append({"image_id": img_id, "category_id": c + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    metrics = {"test/mAP50-95": 0.0, "test/mAP50": 0.0, "test/AR100": 0.0} | {f"test/AP/{c}": 0.0 for c in CLASSES}
    if dets:  # an undertrained model can output nothing, and loadRes([]) crashes
        E = COCOeval(gt, gt.loadRes(dets), "bbox")
        E.evaluate(), E.accumulate(), E.summarize()
        metrics |= {"test/mAP50-95": E.stats[0], "test/mAP50": E.stats[1], "test/AR100": E.stats[8]}
        prec = E.eval["precision"][:, :, :, 0, -1]  # [IoU thr, recall, class, area=all, maxDets=100]
        for k, name in enumerate(CLASSES):
            p = prec[:, :, k]
            metrics[f"test/AP/{name}"] = p[p > -1].mean() if (p > -1).any() else 0.0
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    mlflow.log_metrics(metrics)

    # eyeball check: first test batch, ground truth vs predictions
    imgs, y_true = next(iter(dataset("test")))
    y_pred = {k: v[: len(imgs)] for k, v in pred.items()}
    keras.visualization.plot_bounding_box_gallery(
        imgs, bounding_box_format="xyxy", y_true=y_true, y_pred=y_pred, scale=3,  # grid sized from batch
        value_range=(0, 255), class_mapping=dict(enumerate(CLASSES)), path=str(REPORTS / "test_predictions.png"))
    mlflow.log_artifact(str(REPORTS / "test_predictions.png"))

    # --- export: one graph that takes raw 0-255 images and returns final boxes (no Keras needed to serve) ---
    export = keras.export.ExportArchive()
    export.track(model)

    def serve(images):
        x = model.preprocessor(images)
        return model.decode_predictions(model(x, training=False), x)

    export.add_endpoint("serve", serve, input_signature=[tf.TensorSpec([None, SIZE, SIZE, 3], tf.float32)])
    export.write_out(str(MODELS / "serving"))
    (MODELS / "serving" / "meta.json").write_text(json.dumps({"classes": CLASSES, "image_size": SIZE}))
    mlflow.log_artifacts(str(MODELS / "serving"), "serving")
    print(json.dumps(metrics, indent=2))
