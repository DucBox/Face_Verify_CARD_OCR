import cv2
import numpy as np
from ultralytics import YOLO
from src.utils.utils import detect_objects, load_YOLO, apply_nms
from src.utils.config import FACE_CARD_DETECT_MODEL
from src.utils.image_processing import crop_image, read_image

def detect_face_card(image: np.ndarray, model) -> np.ndarray:
    raw_results = detect_objects(image, model)
    
    final_results = apply_nms(raw_results, iou_threshold = 0.5)
    
    bbox = final_results["face"]
    
    card_face_region = crop_image(image, bbox)
    
    return card_face_region

