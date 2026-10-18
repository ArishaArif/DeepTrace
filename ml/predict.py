import os, numpy as np, torch, joblib, open_clip
from PIL import Image

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_HERE = os.path.dirname(os.path.abspath(__file__))

# Loaded once at import time, not per request
_model, _, _preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
_model = _model.to(_DEVICE).eval()
_clf = joblib.load(os.path.join(_HERE, "deep_clf.joblib"))

# Your teammate's function. Until it exists, we fall back (see predict_image).
try:
    from ml.fractal import predict_fractal      # adjust the path to your repo layout
except ImportError:
    predict_fractal = None

def _load(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")

@torch.no_grad()
def predict_deep(image):
    x = _preprocess(_load(image)).unsqueeze(0).to(_DEVICE)
    v = _model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    return float(_clf.predict_proba(v.cpu().numpy())[0, 1])    # P(fake)

def _placeholder_patches(img, score):
    # TEMPORARY: right shape per the contract, but carries no information
    w, h = img.size
    if min(w, h) < 128:
        s = 128 / min(w, h)
        w, h = round(w * s), round(h * s)
    rows, cols = (h - 64) // 32 + 1, (w - 64) // 32 + 1
    return [[score] * cols for _ in range(rows)]

def predict_image(image):
    img = _load(image)
    deep = predict_deep(img)
    if predict_fractal is None:
        return {"score": deep, "patch_scores": _placeholder_patches(img, deep)}
    fr = predict_fractal(img)
    # Fusion goes here once the fractal branch exists (next step)
    return {"score": deep, "patch_scores": fr["patch_scores"]}
