# Machine Learning Module: Deep Branch & Fusion

This module houses the deep learning branch, feature extraction pipelines, and fusion inference logic for the deepfake and AI-generated media detection platform. It utilizes a frozen **CLIP (`ViT-B/32`)** model combined with a lightweight **Logistic Regression** classifier to classify images and video frames.

---

## 📂 Directory Structure

```text
ml/
├── __init__.py          # Package initializer enabling cross-module imports
├── predict.py           # Core image classification and model fusion logic
├── video_utils.py       # Video frame extraction and face-cropping pipeline
└── deep_clf.joblib      # Trained logistic regression classifier weights

Core Components & API Contract
1. Image Prediction (ml/predict.py)
Exposes the core function required by the backend to process and score individual images.

predict_image(image_path_or_pil):

Input: Accepts either a file path string or an active PIL Image.

Output: Returns a dictionary containing the final score, verdict label, patch score grid (from the fractal branch integration), and heatmap metadata.

2. Video Processing (ml/video_utils.py)
Exposes the utility to handle video inputs by sampling frames and isolating facial regions.

sample_frames(path, every_sec=1.0, max_frames=30, margin=0.3):

Input: A path to a video file.

Output: A list of timestamp-and-crop tuples (t, PIL.Image) representing detected faces sampled at regular intervals (defaults to 1 frame per second).

Model Performance & Specifications
Architecture: Frozen OpenAI CLIP (ViT-B-32) feature extractor + Scikit-Learn Logistic Regression classifier.

Validation Performance: Achieved 93.6% Accuracy and 0.984 AUC on evaluation sets.

Label Convention:

0 = Real

1 = Fake / AI-Generated

Installation & Dependencies
To ensure version compatibility with the saved model weights (.joblib), install the exact pinned package versions listed in the repository root's requirements-DeepBranch.txt:

pip install -r requirements-ml.txt