import numpy as np
import faiss
from typing import List, Tuple
from src.database.firebase_connection import db
from src.utils.config import COLLECTION_NAME

def get_user_embeddings(user_id: str) -> np.ndarray:
    """
    Fetch all face embeddings of a specific user from Firebase.

    Args:
        user_id (str): User ID to fetch.

    Returns:
        np.ndarray: Embeddings array shape (N, 512)
    """
    embeddings = []
    frames_ref = db.collection(COLLECTION_NAME).document(user_id).collection("frames").stream()

    for frame_doc in frames_ref:
        data = frame_doc.to_dict()
        embedding = data.get("embedding", None)
        if embedding:
            embeddings.append(embedding)

    embeddings = np.array(embeddings).astype("float32")
    return embeddings


def verify_face(
    query_embedding: np.ndarray,
    user_embeddings: np.ndarray,
    threshold: float = 0.65
) -> bool:
    """
    Verify if query face matches the given user_id embeddings.

    Args:
        query_embedding (np.ndarray): Embedding to verify (512,).
        user_embeddings (np.ndarray): User's saved embeddings (N, 512).
        threshold (float): Verification threshold.

    Returns:
        bool: True if verified, False otherwise.
    """
    index = faiss.IndexFlatL2(512)
    index.add(user_embeddings)

    D, I = index.search(query_embedding[np.newaxis, :], k=1)
    distance = D[0][0]

    print(f"[INFO] Verification distance: {distance:.4f}")

    return distance < threshold
