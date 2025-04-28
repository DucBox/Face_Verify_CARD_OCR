import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_YOLO(model_path: str) -> YOLO:
    model = YOLO(model_path).to(device)
    return model

def detect_objects(image: np.ndarray, model: YOLO) -> List[Tuple[Tuple[int, int, int, int], float, str]]:
    """
    Detect objects (e.g., CCCD corners, text areas) in an image using a YOLO model.

    Args:
        image (np.ndarray): Input image in numpy array format.
        model (YOLO): Preloaded YOLO model instance.

    Returns:
        List[Tuple[Tuple[int, int, int, int], float, str]]: 
            A list of detected objects. Each item contains:
            - Bounding box coordinates (x1, y1, x2, y2),
            - Confidence score (float),
            - Predicted label (str).
    """
    results = model(image)

    # Initialize list to hold detection results
    detections = []

    for box in results[0].boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = (int(x1), int(y1), int(x2), int(y2))

        # Extract confidence score
        confidence = float(box.conf[0])

        # Extract predicted class label
        label_idx = int(box.cls[0])
        label = model.names[label_idx]

        detections.append((bbox, confidence, label))
    return detections

def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # Compute areas of individual boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute IoU
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou

def apply_nms(
    detections: List[Tuple[Tuple[int, int, int, int], float, str]], 
    iou_threshold: float = 0.5
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Apply NMS on detections.
    Args:
        detections: List of (bbox, confidence, label).
        iou_threshold: Threshold to filter overlapping boxes.
    Returns:
        Dict {label: bbox} after NMS.
    """

    filtered_detections = {}

    # Get all unique labels
    unique_labels = set(label for _, _, label in detections)

    for label in unique_labels:
        label_detections = [det for det in detections if det[2] == label]

        # Sort by confidence descending
        label_detections.sort(key=lambda x: x[1], reverse=True)

        selected_boxes = []

        while label_detections:
            best_box = label_detections.pop(0)
            selected_boxes.append(best_box)

            # Remove boxes with high IoU overlap
            label_detections = [
                box for box in label_detections
                if compute_iou(best_box[0], box[0]) < iou_threshold
            ]

        if selected_boxes:
            filtered_detections[label] = selected_boxes[0][0]

    return filtered_detections

