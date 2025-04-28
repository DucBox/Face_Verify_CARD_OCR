import numpy as np
from src.mini_pipeline.card_process_pipeline import card_process
from src.card_processing.detect_face_card import detect_face_card
from src.card_processing.detect_head import head_detect
from src.face_processing.face_embedding import embed_facenet

def card2face_embedding(image: np.ndarray, detect_face_card_model, head_model) -> np.ndarray:
    #Step 1: Detect face card and crop
    cropped_face = detect_face_card(image = image, model = detect_face_card_model)
    
    #Step 2: Detect head
    face = head_detect(cropped_face, model = head_model)
    
    #Step 3: Embed face
    face_embedding = embed_facenet(face_image = face)
    
    return face_embedding
    