import numpy as np

from src.utils.utils import load_YOLO
from src.utils.config import CARD_DETECT_MODEL
from src.utils.utils import detect_objects, apply_nms, load_YOLO

def detect_card(image: np.ndarray, model) -> dict:
    """
    Detect 4 corners of the card using YOLO model.

    Args:
        image (np.ndarray): Input card image.
        model_path (str): Path to YOLO model.

    Returns:
        dict: Dictionary containing 4 corners with their (x, y) coordinates.
    """

    raw_detections = detect_objects(image, model)

    final_detections = apply_nms(raw_detections, iou_threshold=0.5)

    corners = {}
    required_labels = ["top_left", "top_right", "bottom_left", "bottom_right"]

    for label in required_labels:
        if label in final_detections:
            bbox = final_detections[label]  # (xmin, ymin, xmax, ymax)
            x_center = int((bbox[0] + bbox[2]) / 2)
            y_center = int((bbox[1] + bbox[3]) / 2)
            corners[label] = (x_center, y_center)
        else:
            print(f"⚠️ [WARNING] Missing corner: {label}")

    return corners
