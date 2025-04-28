import numpy as np
import cv2
from src.card_processing.detect_card import detect_card
from src.card_processing.transform_card import transform_perspective
from src.utils.image_processing import read_image

def card_process(image_path: str) -> np.ndarray:
    
    image = read_image(image_path)
    
    #Step 1: Detect 4 corners
    corners = detect_card(image)
    
    #Step 2: Tranform Card Persepctively
    transformed_card = transform_perspective(image, corners)
    
    return transformed_card