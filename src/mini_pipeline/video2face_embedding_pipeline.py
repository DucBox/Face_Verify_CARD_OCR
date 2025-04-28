import uuid

from src.database.store_embedding2firebase import store_embedding_to_db
from src.face_processing.face_embedding import load_facenet, embed_facenet
from src.utils.image_processing import read_video, extract_frames

def video2face_embedding(video_path: str, user_id: str) -> None:
    """
    Pipeline: Extract frames from video, embed faces, and store embeddings.

    Args:
        video_path (str): Path to the input video.
        user_id (str): User identifier to group embeddings.
    """
    # Step 1: Load FaceNet model
    facenet_model = load_facenet()

    # Step 2: Read video
    video = read_video(video_path)

    # Step 3: Extract frames
    frames = extract_frames(video)

    # Step 4: Loop through frames
    for idx, frame in enumerate(frames):
        try:
            # Step 5: Embed face
            face_embedding = embed_facenet(frame, facenet_model)

            # Step 6: Generate unique frame_id
            frame_id = str(uuid.uuid4())

            # Step 7: Store embedding to database
            store_embedding_to_db(face_embedding, user_id, frame_id)

        except Exception as e:
            print(f"[ERROR] Failed to process frame {idx}: {e}")

    print(f"[INFO] Finished processing video for user {user_id}")