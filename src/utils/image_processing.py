import cv2
import numpy as np
import torchvision.transforms as transforms
from typing import List, Tuple
from PIL import Image

def read_image(image_path: str) -> np.ndarray:  
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    cropped = image[ymin:ymax, xmin:xmax]
    return cropped

def read_video(video_path: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(video_path)

def extract_frames(video: cv2.VideoCapture) -> List[np.ndarray]:
    """
    Extract first, middle, and last frame of every second from a video.

    Args:
        video (cv2.VideoCapture): Opened video capture object.

    Returns:
        List[np.ndarray]: List of extracted frames.
    """
    if not video.isOpened():
        print("[ERROR] Cannot open video.")
        return []

    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"[INFO] FPS: {fps}, Total frames: {frame_count}, Duration: {duration:.2f} seconds")

    # Define positions to extract within each second
    frame_selection = [0, fps // 2, fps - 1]

    frames = []
    frame_idx = 0

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        sec = frame_idx // fps
        frame_pos = frame_idx % fps

        if frame_pos in frame_selection:
            frames.append(frame)

        frame_idx += 1

    video.release()
    print(f"[INFO] Extracted {len(frames)} frames.")
    return frames

def preprocess_face_img(face_image):
    """
    Preprocess the face image before feeding it into FaceNet.

    Args:
    face_image (PIL.Image): Face image.

    Returns:
    torch.Tensor: Normalized image tensor.

    """
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(face_image).unsqueeze(0)

def image2pil(image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image (np.ndarray) to PIL Image.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    return pil_image