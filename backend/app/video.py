import cv2
import tempfile
import os
from typing import List, Tuple

def extract_frames(video_path: str, fps_sample_rate: float = 1.0) -> List[Tuple[float, str]]:
    """
    Extracts frames from a video at a specified sampling rate.
    Returns a list of tuples containing (timestamp_in_seconds, frame_temp_filepath).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # Fallback default FPS

    frame_interval = int(video_fps / fps_sample_rate) if video_fps >= fps_sample_rate else 1
    extracted_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps
            
            # Save extracted frame into a temporary JPG file
            tmp_frame = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(tmp_frame.name, frame)
            tmp_frame.close()
            
            extracted_frames.append((timestamp, tmp_frame.name))

        frame_count += 1

    cap.release()
    return extracted_frames

def cleanup_frames(frame_paths: List[Tuple[float, str]]) -> None:
    """Utility function to clean up temporary frame files."""
    for _, path in frame_paths:
        if os.path.exists(path):
            os.remove(path)