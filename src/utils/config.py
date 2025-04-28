from pathlib import Path
import os

# === Base project root ===
BASE_DIR = Path(__file__).resolve().parents[2]

COLLECTION_NAME = "user_face_embeddings"

MODEL_PATH = BASE_DIR / "models"
CARD_DETECT_MODEL = MODEL_PATH / "card_detect.pt" 
FACE_CARD_DETECT_MODEL = MODEL_PATH / "face_card_detect.pt"
HEAD_DETECT_MODEL = MODEL_PATH / "head_detect.pt"
TEXT_DETECT_MODEL = MODEL_PATH / "text_recog.pt"
TEXT_RECOG_MODEL = MODEL_PATH / "transformerocr.pth"