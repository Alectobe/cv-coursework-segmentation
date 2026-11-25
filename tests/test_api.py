import base64
import importlib
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def jpeg_bytes(width=500, height=400):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(checkpoint_path):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(checkpoint_path))
    import api.main

    importlib.reload(api.main)
    with TestClient(api.main.app) as c:
        yield c
    mp.undo()


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"]["experiment"] == "attn_gates"
    assert body["model"]["clearml_task_id"] == "test-task"


def test_predict_json(client):
    r = client.post("/predict", files={"file": ("polyp.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert (body["width"], body["height"]) == (500, 400)
    assert 0 <= body["polyp_area_ratio"] <= 1
    mask = Image.open(io.BytesIO(base64.b64decode(body["mask_png_base64"])))
    assert mask.size == (500, 400)
    assert set(np.unique(np.asarray(mask))) <= {0, 255}
    assert body["overlay_png_base64"]


def test_predict_threshold_changes_mask(client):
    data = jpeg_bytes()
    low = client.post("/predict?threshold=0.0", files={"file": ("a.jpg", data, "image/jpeg")}).json()
    high = client.post("/predict?threshold=1.0", files={"file": ("a.jpg", data, "image/jpeg")}).json()
    assert low["polyp_area_ratio"] >= high["polyp_area_ratio"]
    assert high["polyp_found"] is False


def test_predict_png_endpoints(client):
    for endpoint in ("/predict/mask", "/predict/overlay"):
        r = client.post(endpoint, files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(r.content)).size == (500, 400)


@pytest.mark.parametrize(
    "payload, status",
    [
        (("notes.txt", b"definitely not an image", "text/plain"), 400),
        (("empty.jpg", b"", "image/jpeg"), 400),
        (("broken.jpg", jpeg_bytes()[:200], "image/jpeg"), 400),
        (("tiny.jpg", jpeg_bytes(10, 10), "image/jpeg"), 400),
    ],
)
def test_bad_files(client, payload, status):
    r = client.post("/predict", files={"file": payload})
    assert r.status_code == status
    assert r.json()["detail"]


def test_missing_file_and_bad_threshold(client):
    assert client.post("/predict").status_code == 422
    r = client.post("/predict?threshold=2", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 422


def test_health_503_without_weights(tmp_path):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    import api.main

    importlib.reload(api.main)
    with TestClient(api.main.app) as c:
        r = c.get("/health")
        assert r.status_code == 503
        assert "missing.pt" in r.json()["detail"]
        assert c.post("/predict", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}).status_code == 503
    mp.undo()
