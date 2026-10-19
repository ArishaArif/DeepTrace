import os

# Shared contract default. DEEP BRANCH tunes it on the validation set only; set it at deploy
# time with the THRESHOLD environment variable, no code change needed.
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

MAX_IMAGE_MB = 10
MAX_VIDEO_MB = 50
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv"}

# Day 3 hardening (these do not change the API contract)
MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "30"))       # frames actually scored, evenly spread
VIDEO_TIME_BUDGET_S = float(os.getenv("VIDEO_TIME_BUDGET_S", "60"))  # stop scoring after this; return partial timeline
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))  # model calls running at once; extra requests wait
SLOW_WARN_S = float(os.getenv("SLOW_WARN_S", "8"))                # log a warning when a request takes longer
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
