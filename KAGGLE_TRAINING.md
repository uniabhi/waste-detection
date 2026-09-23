# Training on Kaggle GPU: step by step

Your laptop has no GPU, so Kaggle runs the training. The code stays in GitHub and the data lives in a Kaggle dataset.
`notebooks/kaggle_train.ipynb` just clones the repo and runs `dvc repro`, which runs auto-label → split → train.

```
GitHub repo (code) ──git clone──┐
                                ├──► Kaggle notebook (GPU T4 x2) ──► outputs.zip (model, metrics, MLflow runs)
Kaggle dataset (images) ──mount─┘
```

**Kaggle limits to plan around:**
- **Weekly GPU quota:** about 30 GPU hours. Check it on your Kaggle profile under Settings.
- **Session length:** at most 12 hours per session.
- **CPU and RAM:** 4 CPU cores and about 30 GB RAM.
- **Accelerator:** use **GPU T4 x2**, not TPU. This pipeline isn't built for TPU.

(Kaggle occasionally renames buttons. If a label below doesn't match exactly, look for the closest one.)

---

## Step 0: One-time account setup

1. Create an account on [kaggle.com](https://www.kaggle.com).
2. **Verify your phone number** (Settings → Phone verification). Without it, Kaggle won't let you turn on GPU or Internet.

## Step 1: Put the code on GitHub

From the project folder on your laptop:

```bash
git init
dvc init                                  # optional but recommended; the notebook falls back if you skip it
git add .
git commit -m "waste detection project"
git branch -M main
git remote add origin https://github.com/<you>/waste-detection.git
git push -u origin main
```

`archive/` is in `.gitignore`, so the images stay off GitHub and go to Kaggle in Step 2.

**Private repo?** Create a GitHub personal access token with read access. In the notebook, set
`REPO = "https://<token>@github.com/<you>/waste-detection.git"`. Better still, store the token as a Kaggle Secret (Step 4).

## Step 2: Upload the images as a Kaggle dataset (once)

1. Zip only the folder you need: right-click `archive/standardized_384` → Compress to ZIP (about 450 MB).
2. On Kaggle, go to **Datasets → New Dataset**, upload the zip, name it (for example `waste-384`), and set it to **Private**. Wait until processing finishes.

If your dataset is already public on Kaggle, skip this and attach the public one in Step 3.

## Step 3: Create the notebook

1. Go to **Code → New Notebook**, then **File → Import Notebook** and upload `notebooks/kaggle_train.ipynb`.
2. Open the notebook's right-hand panel (**Settings / Session options**):
   - **Accelerator:** `GPU T4 x2`
   - **Internet:** `On` (needed for `pip install`, `git clone`, and downloading pretrained weights)
3. Click **Add Input**, search **Your Datasets**, and add the dataset from Step 2.
   It appears under `/kaggle/input/...`, and the notebook finds the `standardized_384` folder automatically.
4. In the first code cell, set `REPO` to your GitHub URL.

## Step 4 (optional): Secrets for remote storage and tracking

Go to **Add-ons → Secrets**, add the ones you want, and tick "attach to this notebook":

| Secret | What it enables |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | DVC push/pull to your S3 remote. Results survive the session, and later runs skip stages that already ran, such as the 30-min auto-labeling. |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | Live MLflow tracking on a server (DagsHub gives one free) instead of a local `mlruns/` folder. |

Without secrets everything still works. You just download results as a zip at the end.

## Step 5: Smoke test first (about 10 minutes of GPU time)

This proves the whole pipeline works before you spend hours of quota.

1. Leave `SMOKE_TEST = True` in the notebook's second code cell. That uses 20 images per class and 1 epoch.
2. Click **Run All** and watch the cells.
3. It passed if:
   - the first cell prints a `T4` GPU line and TensorFlow lists 2 GPUs
   - `dvc repro` runs `autolabel`, then `split`, then `train` with no traceback
   - the last cell prints `reports/autolabel.json` (images labeled per class) and `reports/metrics.json` (mAP numbers)
4. **Expect low mAP.** One epoch on 200 images is meant to prove the code runs, not to produce a good model.
5. **Note how long one training epoch took.** You'll use it to size the real run.

Got an error? Check Troubleshooting below, or copy the traceback and ask.

## Step 6: The real training run

1. Set `SMOKE_TEST = False`.
2. Adjust `train.epochs` in `params.yaml` if needed. Rough budget: 30 min of auto-labeling + (seconds per epoch × epochs) must fit comfortably under 12 hours. Early stopping (`patience`) usually ends training sooner anyway.
3. Click **Save Version → Save & Run All (Commit)**.
   This runs in the background. You can close the browser or even your laptop.
4. Follow progress in **View Active Events** (bottom left), or open the running version's log.

## Step 7: Get the results

When the version finishes, open it and download `outputs.zip` from the **Output** section. It contains:

| Path | What it is |
|---|---|
| `reports/metrics.json` | Test-set mAP@[.5:.95], mAP@.5, recall, per-class AP |
| `reports/test_predictions.png` | Ground truth vs predicted boxes. **Look at this first.** |
| `reports/autolabel.json` | How many images the auto-labeler kept per class |
| `models/serving/` | TF SavedModel, which is what the FastAPI app and Docker image serve |
| `models/detector.keras` | Full Keras model, for fine-tuning further |
| `data/` | Auto-labels + COCO splits, so you don't redo the labeling |
| `mlruns/` | MLflow runs (view locally with `mlflow ui`) |
| `dvc.lock` | Exact hashes of data, params, and outputs for this run |

Then back on your laptop, do one of these:
- **With S3 secrets:** copy `dvc.lock` and `reports/` from the zip into the repo, then run `dvc pull` and `git commit -am "trained model"`.
- **Without them:** unzip everything into the repo, then run `dvc commit` and `git commit -am "trained model"`.

Test the model locally: `uvicorn app.main:app` and open http://localhost:8000/docs.

## Step 8: Next experiments

1. Change one thing in `params.yaml` (`learning_rate`, `epochs`, or `box_threshold` for the labeler) and push.
2. On Kaggle, open the notebook → **Edit** → Save & Run All again.
3. Compare runs with `dvc metrics diff` or in MLflow.

With S3 secrets set, changing only `train.*` skips the 30-min auto-labeling, because DVC restores it from the run-cache.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| No GPU listed / `nvidia-smi` not found | Accelerator isn't set to GPU T4 x2, or your phone isn't verified. |
| `pip install` or `git clone` fails | Internet is Off in the notebook settings. |
| `cp: cannot stat ''` in the data cell | Dataset not attached (Add Input), or the folder inside isn't named `standardized_384`. |
| `ResourceExhaustedError` / OOM | Lower `train.batch_size` in `params.yaml` to 8. |
| Error mentioning `MirroredStrategy` / NCCL / multi-GPU | Train on one GPU: add `import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"` at the very top of the first cell. |
| `AssertionError: ... expected 384x384` in split | An image isn't 384×384. Use `standardized_384`, not `original`. |
| Run hit the 12 h limit | A timed-out run may not keep its outputs. Size `epochs` from the smoke test's time per epoch, with margin. |
| Quota running low | Develop with `SMOKE_TEST = True`, and do full runs only when something actually changed. |
