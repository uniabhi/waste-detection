# Waste detection: end-to-end object detection with TensorFlow/Keras + MLOps

This project detects 10 kinds of waste (battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash) and serves the detector as a REST API.

**Stack:**
- **Model:** TensorFlow / Keras 3 / KerasHub RetinaNet.
- **Training:** Kaggle GPU.
- **Data versioning and pipeline:** DVC.
- **Experiment tracking:** MLflow.
- **Evaluation:** COCO mAP via `pycocotools`.
- **Serving:** FastAPI + TF SavedModel.
- **Deployment:** Docker, Kubernetes, and GitHub Actions → ECR → EKS.

## Read first: the dataset has no boxes

`archive/` is a **classification** dataset: the folder name is the class, and there are no bounding boxes.
Stage 1 creates the boxes. It prompts **Grounding DINO**, a zero-shot text-prompted detector, with each folder's class name, and every confident box becomes a label.
This is called auto-labeling (or distillation): a big, slow foundation model labels the data, then a smaller detector learns from those labels.
At work, the next step would be a human reviewing a sample of the labels in CVAT or Label Studio.

`src/autolabel.py` is the only PyTorch file, because Grounding DINO has no TensorFlow implementation. It's a labeling tool that runs once. The model you train, evaluate, export, and serve is pure TF/Keras.

Limitation: most images show one centered object, so expect weaker results on cluttered real-world scenes.
You can switch to a truly annotated dataset later (for example TACO, which ships COCO JSON). Drop the `autolabel` stage and point `split.py` at the real labels. Nothing downstream changes.

## Pipeline

```
archive/standardized_384 ─autolabel─► data/autolabels.json ─split─► data/coco/{train,val,test}.json ─train─► models/detector.keras
    (DVC-tracked)      Grounding DINO                    stratified 80/10/10, COCO format   RetinaNet  models/serving (SavedModel)
                                                                                                  │
                                              MLflow: params, per-epoch losses, test mAP, prediction grid
                                                                                                  └─► FastAPI ─► Docker ─► ECR ─► EKS
```

`dvc.yaml` wires the stages together, and `params.yaml` holds every knob. `dvc repro` reruns only the stages whose code, data, or params changed.
For example, changing `train.*` retrains the model without redoing the 30-minute auto-labeling.

## Where each detection concept lives

| Concept | File |
|---|---|
| Box formats: xyxy (Keras), COCO `[x, y, w, h]`, pixel vs normalized | `src/autolabel.py`, `src/split.py` |
| Zero-shot detection, pseudo-labels, why unlabeled images ≠ background | `src/autolabel.py` |
| COCO JSON structure, stratified split | `src/split.py` |
| `tf.data` for detection: ragged boxes, padding with label -1, batching | `src/train.py` `dataset()` |
| Box-aware augmentation (flip moves boxes too) | `src/train.py` `random_flip()` |
| Transfer learning, anchors + FPN + focal loss (RetinaNet), multi-GPU `MirroredStrategy` | `src/train.py` |
| NMS and confidence thresholds (prediction decoder) | `src/train.py` |
| mAP@[.5:.95], mAP@.5, recall, per-class AP with `COCOeval` | `src/train.py` |
| SavedModel export with a custom serving signature | `src/train.py` |
| Letterbox preprocessing, mapping boxes back to the original image | `app/detector.py` |
| Serving, input validation, EXIF rotation | `app/main.py` |

After training, open `reports/test_predictions.png`, which shows ground truth vs predictions on test images. It's the quickest way to spot bad labels or a bad model.

## 1. One-time setup (local)

```bash
git init && dvc init
dvc remote add -d storage s3://<your-bucket>/waste-detection   # DVC remote (DagsHub also works, free)
dvc add archive/standardized_384 && dvc push                   # version the raw data
git add . && git commit -m "waste detection project" && git push
```

## 2. Train on Kaggle (GPU)

Follow **[KAGGLE_TRAINING.md](KAGGLE_TRAINING.md)**. It covers account setup, uploading the data, the notebook settings, a 10-minute smoke test, the real run, downloading results, and troubleshooting.

## 3. Serve

```bash
pip install -r requirements-serve.txt
uvicorn app.main:app --reload                   # Swagger UI at http://localhost:8000/docs
curl -F "file=@some_photo.jpg" "localhost:8000/predict?conf=0.3"

docker build -t waste-detector . && docker run -p 8000:8000 waste-detector

# local Kubernetes (kind)
kind load docker-image waste-detector:latest
kubectl apply -f k8s/ && kubectl port-forward svc/waste-detector 8000:80
```

## 4. CI/CD (`.github/workflows/ci.yml`)

- **Every push:** runs `ruff` and `pytest`.
- **On `main`:** runs once you set these:
  - repo variables `ECR_REPOSITORY`, `AWS_REGION`, `EKS_CLUSTER`
  - secret `AWS_ROLE_ARN` (an IAM role for GitHub OIDC)

  The main-branch job then:
  1. `dvc pull`s the exact SavedModel that `dvc.lock` pins in git.
  2. Tests the API against it.
  3. Builds the image and pushes it to ECR.
  4. Rolls it out to EKS.

## The daily loop (what the office job looks like)

1. Change `params.yaml` (for example `learning_rate`, or `box_threshold` for the auto-labeler) and commit.
2. Run it on Kaggle.
3. Compare the result with `dvc metrics diff` or in MLflow.
4. Keep the change or revert it.
5. Merge to `main`, and the new model is deployed.

## Tests

```bash
python -m pytest -q      # letterbox math always; the API test once models/serving exists
```
