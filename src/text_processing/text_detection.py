from src.utils.config import TEXT_DETECT_MODEL
from src.utils.utils import load_YOLO, apply_nms, detect_objects
from src.utils.image_processing import read_image
from typing import Dict
import numpy as np

def filter_text_boxes(detections: Dict[str, tuple]) -> Dict[str, tuple]:

    valid_labels = {"id", "name", "birth"}
    filtered_boxes = {label: bbox for label, bbox in detections.items() if label in valid_labels}

    print(f"[INFO] Filtered text fields: {filtered_boxes}")
    return filtered_boxes

def detect_text(image: np.ndarray, model_path: str = TEXT_DETECT_MODEL):
    
    model = load_YOLO(TEXT_DETECT_MODEL)
    
    raw_results = detect_objects(image, model)
    
    final_results = apply_nms(raw_results)
    
    filtered_results = filter_text_boxes(final_results)
    
    return filtered_results