import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from src.utils.image_processing import image2pil, preprocess_face_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_facenet() -> InceptionResnetV1:
    """
    Load the FaceNet model (pretrained on VGGFace2).
    """
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model

def embed_facenet(face_image: np.ndarray, facenet_model = load_facenet()) -> np.ndarray:
    """
    Generate a 512-dimensional feature vector from a face image.
    """
    pil_image = image2pil(face_image)
    face_tensor = preprocess_face_img(pil_image).to(device)

    with torch.no_grad():
        embedding = facenet_model(face_tensor).cpu().numpy()

    return embedding.squeeze()
