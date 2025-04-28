import numpy as np
import cv2
from src.utils.utils import detect_objects, load_YOLO, apply_nms
from src.utils.image_processing import crop_image, read_image
from src.utils.config import HEAD_DETECT_MODEL
from src.card_processing.detect_face_card import detect_face_card

def head_detect(image: np.ndarray, model) -> np.ndarray:
    raw_results = detect_objects(image, model)
    
    final_results = apply_nms(raw_results, iou_threshold = 0.5)
    
    bbox = final_results["head"]
    
    head_region = crop_image(image, bbox)
    
    return head_region
