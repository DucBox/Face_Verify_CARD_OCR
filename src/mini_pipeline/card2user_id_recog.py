import numpy as np
from src.mini_pipeline.card2face_embedding_pipeline import card2face_embedding
from src.retrieval.face_retrieval_recog import get_embeddings_from_db, query_embedding

def card2user_id_recog(card_image: np.ndarray) -> str:
    """
    Full mini pipeline: Card image -> Face embedding -> Find matched user_id.

    Args:
        card_image (np.ndarray): Input card image.

    Returns:
        str: Matched user_id.
    """
    # Step 1: Embed face from card
    face_embedding = card2face_embedding(card_image)

    # Step 2: Get all db embeddings
    db_embeddings, user_ids = get_embeddings_from_db()

    # Step 3: Query using FAISS
    matched_user_id = query_embedding(face_embedding, db_embeddings, user_ids)

    return matched_user_id
