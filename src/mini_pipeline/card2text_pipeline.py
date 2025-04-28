import numpy as np
from src.mini_pipeline.card_process_pipeline import card_process
from src.utils.image_processing import read_image, crop_image
from src.text_processing.text_detection import detect_text  
from src.text_processing.text_recognition import recog_text, load_vietocr_model

def card2text(transformed_card_image: np.ndarray, vietocr_model, text_detect_model) -> dict:
    """
    Full pipeline: From card image path -> detect text regions -> recognize text.

    Args:
        card_path (str): Path to the card image.

    Returns:
        dict: Extracted text fields {label: text}.
    """
    print("[INFO] Detecting text regions ...")
    text_boxes = detect_text(transformed_card_image, text_detect_model)

    if not text_boxes:
        print("❌ No valid text regions detected.")
        return {}

    extracted_text = {}
    for label, (x1, y1, x2, y2) in text_boxes.items():
        cropped_text_region = crop_image(transformed_card_image, (x1, y1, x2, y2))
        text = recog_text(cropped_text_region, vietocr_model)
        extracted_text[label] = text
        print(f"[INFO] {label}: {text}")

    return extracted_text
