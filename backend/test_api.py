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


# ---------- Day 2/3 additions ----------
import time as _time

import numpy as np

from app import video as video_mod
from app.heatmap import render_heatmap


def test_timing_header_and_contract_untouched():
    r = client.post("/analyze/image", files={"file": ("t.png", make_image(), "image/png")})
    assert int(r.headers["X-Process-Time-ms"]) >= 0
    assert set(r.json()) == {"verdict", "score", "patch_scores", "heatmap_png"}   # no extra keys


def test_empty_upload_rejected():
    r = client.post("/analyze/image", files={"file": ("e.png", b"", "image/png")})
    assert r.status_code == 400


def test_heatmap_alignment():
    """A hot patch at the top-left must light up the top-left, not the bottom-right."""
    grid = np.zeros((5, 8), dtype=np.float32)
    grid[0, 0] = 1.0
    png = Image.open(io.BytesIO(base64.b64decode(render_heatmap(grid.tolist(), (300, 200)))))
    a = np.array(png)[..., 3]
    assert a[:40, :40].mean() > a[-40:, -40:].mean() + 50


def test_image_oserror_maps_to_400(monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "predict_image", lambda p: (_ for _ in ()).throw(OSError("bad")))
    r = client.post("/analyze/image", files={"file": ("t.png", make_image(), "image/png")})
    assert r.status_code == 400


def test_video_frame_cap_spread(monkeypatch):
    monkeypatch.setattr(video_mod, "DUMMY", False)
    monkeypatch.setattr(video_mod, "sample_frames",
                        lambda p: [(i * 0.5, Image.new("RGB", (128, 128))) for i in range(100)])
    monkeypatch.setattr(video_mod, "predict_image", lambda p: {"score": 0.7})
    out = video_mod.analyze_video("x.mp4")
    ts = [f["t"] for f in out["frames"]]
    assert len(ts) == config.MAX_VIDEO_FRAMES
    assert ts == sorted(ts) and ts[0] == 0.0 and ts[-1] == 49.5      # spread over the whole video
    assert out["verdict"] == "fake"


def test_video_time_budget_returns_partial(monkeypatch):
    monkeypatch.setattr(video_mod, "DUMMY", False)
    monkeypatch.setattr(config, "VIDEO_TIME_BUDGET_S", 0.05)
    monkeypatch.setattr(video_mod, "sample_frames",
                        lambda p: [(float(i), Image.new("RGB", (128, 128))) for i in range(20)])

    def slow(p):
        _time.sleep(0.03)
        return {"score": 0.2}

    monkeypatch.setattr(video_mod, "predict_image", slow)
    out = video_mod.analyze_video("x.mp4")
    assert 1 <= len(out["frames"]) < 20 and out["verdict"] == "real"
