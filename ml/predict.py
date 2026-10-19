import os, logging, numpy as np, torch, joblib, open_clip
from PIL import Image

log = logging.getLogger(__name__)
_DEVICE = "cuda" if torch.torch.cuda.is_available() else "cpu"
_WEIGHTS = os.environ.get("ML_WEIGHTS_DIR",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"))

_model, _, _preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
_model = _model.to(_DEVICE).eval()
_clf = joblib.load(os.path.join(_WEIGHTS, "deep_clf.joblib"))

_FUSION_PATH = os.path.join(_WEIGHTS, "fusion.joblib")
_fusion = joblib.load(_FUSION_PATH) if os.path.exists(_FUSION_PATH) else None

# Correctly import the predict function from the fractal module/file
try:
    from ml.fractal import predict_fractal
    # If predict_fractal was imported as a module package, unwrap it to the function:
    if hasattr(predict_fractal, "predict_fractal"):
        predict_fractal = predict_fractal.predict_fractal
except ImportError:
    predict_fractal = None

def _load(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")

def _logit(p):
    p = float(np.clip(p, 1e-3, 1 - 1e-3))
    return float(np.log(p / (1 - p)))

@torch.no_grad()
def predict_deep(image):
    x = _preprocess(_load(image)).unsqueeze(0).to(_DEVICE)
    v = _model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    return float(_clf.predict_proba(v.cpu().numpy())[0, 1])

def _placeholder_patches(img, score):
    w, h = img.size
    if min(w, h) < 128:
        s = 128 / min(w, h)
        w, h = round(w * s), round(h * s)
    return [[score] * ((w - 64) // 32 + 1) for _ in range((h - 64) // 32 + 1)]

def predict_image(image):
    img = _load(image)
    deep = predict_deep(img)
    if predict_fractal is None or not callable(predict_fractal):
        log.warning("fractal module not callable: PLACEHOLDER heatmap in use")
        return {"score": deep, "patch_scores": _placeholder_patches(img, deep)}
    try:
        fr = predict_fractal(img)
    except FileNotFoundError:
        log.warning("fractal weights missing: PLACEHOLDER heatmap in use")
        return {"score": deep, "patch_scores": _placeholder_patches(img, deep)}
    
    patches = np.asarray(fr["patch_scores"], dtype=float).tolist()
    frac = float(fr["fractal_score"])
    if _fusion is None:
        log.warning("fusion.joblib missing: score is the deep branch only")
        score = deep
    else:
        score = float(_fusion.predict_proba([[_logit(deep), _logit(frac)]])[0, 1])
    return {"score": score, "patch_scores": patches}
