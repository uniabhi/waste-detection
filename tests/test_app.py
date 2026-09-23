import io
import os

import pytest
from PIL import Image

from app.detector import letterbox

MODEL = os.getenv("MODEL_PATH", "models/serving")


def test_letterbox_geometry():
    canvas, r, (px, py) = letterbox(Image.new("RGB", (200, 100)), 64)
    assert canvas.size == (64, 64) and r == 0.32 and (px, py) == (0, 16)
    assert canvas.getpixel((0, 0)) == (114, 114, 114)  # padding
    # a point at letterbox (32, 32) maps back to the original image centre
    assert ((32 - px) / r, (32 - py) / r) == (100, 50)


@pytest.mark.skipif(not os.path.isdir(MODEL), reason="no trained model")
def test_predict_endpoint():
    from fastapi.testclient import TestClient

    from app.main import app

    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (200, 50, 50)).save(buf, "JPEG")
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/predict", files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")})
        assert r.status_code == 200 and isinstance(r.json()["detections"], list)
        for d in r.json()["detections"]:
            x1, y1, x2, y2 = d["box"]
            assert 0 <= x1 <= x2 <= 320 and 0 <= y1 <= y2 <= 240
        assert client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
