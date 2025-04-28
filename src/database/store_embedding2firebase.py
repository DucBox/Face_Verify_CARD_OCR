from typing import List
import numpy as np
from src.database.firebase_connection import db
from src.utils.config import COLLECTION_NAME

def store_embedding_to_db(embedding: np.ndarray, user_id: str, frame_id: str) -> None:
    """
    Store face embedding vector into Firebase Firestore.

    Args:
        embedding (np.ndarray): Face embedding vector (512,).
        user_id (str): User identifier.
        frame_id (str): Frame identifier.
    """
    embedding_list = embedding.tolist()

    # Define document path: collection 'face_embeddings' -> document 'user_id' -> subcollection 'frames'
    doc_ref = db.collection(COLLECTION_NAME).document(user_id).collection("frames").document(frame_id)

    doc_ref.set({
        "embedding": embedding_list
    })

    print(f"[INFO] Stored embedding for user: {user_id}, frame: {frame_id}")
