import numpy as np
import faiss
from typing import List, Tuple
from src.database.firebase_connection import db
from src.utils.config import COLLECTION_NAME

def get_embeddings_from_db() -> Tuple[np.ndarray, List[str]]:
    """
    Fetch all stored face embeddings from Firebase.

    Returns:
        Tuple: (embeddings array shape (N, 512), list of user_ids)
    """
    user_ids = []
    embeddings = []

    users_ref = db.collection(COLLECTION_NAME).stream()

    for user_doc in users_ref:
        user_id = user_doc.id
        frames_ref = db.collection(COLLECTION_NAME).document(user_id).collection("frames").stream()
        
        for frame_doc in frames_ref:
            data = frame_doc.to_dict()
            embedding = data.get("embedding", None)
            if embedding:
                embeddings.append(embedding)
                user_ids.append(user_id)

    embeddings = np.array(embeddings).astype("float32")
    return embeddings, user_ids


def query_embedding(embedding: np.ndarray, db_embeddings: np.ndarray, user_ids: List[str], top_k: int = 1) -> str:
    """
    Query the database embeddings to find the closest user.

    Args:
        embedding (np.ndarray): Query embedding vector (512,).
        db_embeddings (np.ndarray): All stored embeddings (N, 512).
        user_ids (List[str]): List of user_ids corresponding to db_embeddings.
        top_k (int): How many top matches to retrieve.

    Returns:
        str: Matched user_id (highest similarity).
    """
    # Build FAISS index
    index = faiss.IndexFlatL2(512)  # 512 dimensions
    index.add(db_embeddings)

    # Search
    D, I = index.search(embedding[np.newaxis, :], top_k)

    top_idx = I[0][0]
    matched_user_id = user_ids[top_idx]

    return matched_user_id
