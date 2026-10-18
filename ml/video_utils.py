import cv2
from PIL import Image

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _largest_face(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])

def sample_frames(path, every_sec=1.0, max_frames=30, margin=0.3):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("unreadable video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every_sec)))
    results, idx = [], 0
    while len(results) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            box = _largest_face(frame)
            if box is not None:
                x, y, w, h = box
                m = int(margin * max(w, h))
                H, W = frame.shape[:2]
                crop = frame[max(0, y - m):min(H, y + h + m),
                             max(0, x - m):min(W, x + w + m)]
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                results.append((idx / fps, img))
        idx += 1
    cap.release()
    return results
