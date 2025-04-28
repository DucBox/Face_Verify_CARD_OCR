from src.utils.config import TEXT_RECOG_MODEL
from src.utils.image_processing import image2pil
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import streamlit as st
import numpy as np

@st.cache_resource
def load_vietocr_model(model_path: str = TEXT_RECOG_MODEL) -> Predictor:
    """
    Load VietOCR model from the given path.
    """
    config = Cfg.load_config_from_name('vgg_transformer')
    # config['weights'] = str(model_path)
    config['device'] = 'cuda'  

    model = Predictor(config)
    return model

def recog_text(cropped_text_region: np.ndarray, model = Predictor):
    cropped_pil_image = image2pil(cropped_text_region)
    text = model.predict(cropped_pil_image).strip()
    return text