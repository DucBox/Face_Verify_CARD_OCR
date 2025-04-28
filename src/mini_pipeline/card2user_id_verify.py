import numpy as np
from src.mini_pipeline.card2face_embedding_pipeline import card2face_embedding
from src.retrieval.face_retrieval_verify import get_user_embeddings, verify_face

def card2user_id_verify(card_image: np.ndarray) -> str:
    """
    Full mini pipeline: Card image -> Face embedding -> Verify Match or Not.

    Args:
        card_image (np.ndarray): Input card image.

    Returns:
        str: True or False
    """
    # 1. Embed query face (ảnh chụp thẻ chẳng hạn)
    query_embedding = card2face_embedding(card_image)

    # 2. Get all embeddings của user "user_123"
    user_embeddings = get_user_embeddings("user_123")

    # 3. Verify
    is_verified = verify_face(query_embedding, user_embeddings, threshold=0.65)

    if is_verified:
        print("✅ User verified successfully!")
    else:
        print("❌ Verification failed.")

    return is_verified
