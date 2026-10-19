import cv2
import numpy as np
import base64

def generate_heatmap_base64(image_path: str, patch_scores: list) -> str:
    """
    Overlays the patch_scores grid from the Fractal Branch onto the original image
    and returns a base64 encoded PNG data URI.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image for heatmap generation.")
    
    h, w, _ = img.shape
    scores_matrix = np.array(patch_scores, dtype=np.float32)
    
    # Resize patch score grid to match full original image dimensions
    heatmap_resized = cv2.resize(scores_matrix, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize score values (0.0 to 1.0) and map to 0-255 uint8
    heatmap_normalized = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    
    # Apply JET color map (regions contributing to fake score appear warmer)
    color_heatmap = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    # Blend color heatmap onto original image (60% original, 40% heatmap overlay)
    blended = cv2.addWeighted(img, 0.6, color_heatmap, 0.4, 0)
    
    # Encode output to PNG byte buffer
    success, buffer = cv2.imencode('.png', blended)
    if not success:
        raise RuntimeError("Failed to encode heatmap image.")
        
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"