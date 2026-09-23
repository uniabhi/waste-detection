import io
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, ImageOps

from app.detector import Detector

MAX_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector(os.getenv("MODEL_PATH", "models/serving"))  # load once, not per request
    yield


app = FastAPI(title="Waste detector", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: Annotated[UploadFile, File()], conf: Annotated[float, Query(ge=0, le=1)] = 0.25):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))  # phone photos are stored rotated
        img.load()
    except (OSError, Image.DecompressionBombError):
        raise HTTPException(400, "not a valid image")
    return {"width": img.width, "height": img.height, "detections": app.state.detector(img, conf)}
