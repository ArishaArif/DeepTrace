import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def make_image(w=200, h=160):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "gray").save(buf, format="PNG")
    return buf.getvalue()


def test_image_contract():
    r = client.post("/analyze/image", files={"file": ("a.png", make_image(), "image/png")})
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"verdict", "score", "patch_scores", "heatmap_png"}
    assert d["verdict"] in ("real", "fake")
    assert len(d["patch_scores"]) == (160 - 64) // 32 + 1
    assert len(d["patch_scores"][0]) == (200 - 64) // 32 + 1


def test_small_image_is_upscaled_grid():
    r = client.post("/analyze/image", files={"file": ("a.png", make_image(64, 64), "image/png")})
    assert len(r.json()["patch_scores"]) == 3     # 128x128 -> 3x3 patches


def test_video_contract():
    r = client.post("/analyze/video", files={"file": ("a.mp4", b"x" * 100, "video/mp4")})
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"verdict", "score", "frames"}
    assert set(d["frames"][0]) == {"t", "score"}


def test_bad_type():
    r = client.post("/analyze/image", files={"file": ("a.txt", b"hi", "text/plain")})
    assert r.status_code == 400
