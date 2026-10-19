import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.inference import DUMMY
from app.main import app

client = TestClient(app)


def make_image(w=200, h=150):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 80, 200)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def expected_grid(w, h):
    """Contract: images under 128 are resized up; 64x64 patches, stride 32."""
    w, h = max(w, 128), max(h, 128)
    return (h - 64) // 32 + 1, (w - 64) // 32 + 1


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["dummy_mode"], bool)


@pytest.mark.parametrize("w,h", [(200, 150), (64, 64), (500, 333)])
def test_image_contract(w, h):
    r = client.post("/analyze/image", files={"file": ("t.png", make_image(w, h), "image/png")})
    assert r.status_code == 200
    body = r.json()

    assert set(body) == {"verdict", "score", "patch_scores", "heatmap_png"}
    assert body["verdict"] in ("fake", "real")
    assert 0 <= body["score"] <= 1
    assert body["verdict"] == ("fake" if body["score"] >= config.THRESHOLD else "real")

    rows, cols = expected_grid(w, h)
    grid = body["patch_scores"]
    assert len(grid) == rows
    assert all(len(row) == cols for row in grid)
    assert all(0 <= v <= 1 for row in grid for v in row)

    heatmap = Image.open(io.BytesIO(base64.b64decode(body["heatmap_png"])))
    assert heatmap.format == "PNG"
    assert heatmap.mode == "RGBA"
    assert heatmap.size == (w, h)


def test_image_unsupported_type():
    r = client.post("/analyze/image", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_image_unreadable():
    r = client.post("/analyze/image", files={"file": ("broken.png", b"not an image", "image/png")})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_image_too_large():
    big = b"0" * ((config.MAX_IMAGE_MB + 1) * 1024 * 1024)
    r = client.post("/analyze/image", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413
    assert "detail" in r.json()


def test_video_unsupported_type():
    r = client.post("/analyze/video", files={"file": ("clip.gif", b"x", "image/gif")})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_video_too_large():
    big = b"0" * ((config.MAX_VIDEO_MB + 1) * 1024 * 1024)
    r = client.post("/analyze/video", files={"file": ("big.mp4", big, "video/mp4")})
    assert r.status_code == 413
    assert "detail" in r.json()


@pytest.mark.skipif(not DUMMY, reason="dummy video needs no decoder; real mode needs a real video file")
def test_video_contract():
    r = client.post("/analyze/video", files={"file": ("clip.mp4", b"placeholder", "video/mp4")})
    assert r.status_code == 200
    body = r.json()

    assert set(body) == {"verdict", "score", "frames"}
    assert body["verdict"] in ("fake", "real")
    assert 0 <= body["score"] <= 1
    assert body["verdict"] == ("fake" if body["score"] >= config.THRESHOLD else "real")

    frames = body["frames"]
    assert frames, "frames must not be empty"
    assert frames[0]["t"] == 0.0
    for f in frames:
        assert set(f) == {"t", "score"}
        assert 0 <= f["score"] <= 1
    mean = sum(f["score"] for f in frames) / len(frames)
    assert body["score"] == pytest.approx(mean, abs=1e-6)
