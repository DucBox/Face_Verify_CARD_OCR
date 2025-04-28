import cv2
import numpy as np

def transform_perspective(image: np.ndarray, corners: dict, output_size=(800, 500)):
    """
    Transform the perspective of the CCCD image to the standard form.

    Args:
    image (numpy.ndarray): Original image.
    corners (dict): The 4 corner coordinates of the CCCD.
    output_size (tuple): The size of the output image.

    Returns:
    numpy.ndarray: Image after transformation, or None if error.
    """
    try:
        if len(corners) != 4:
            raise ValueError("⚠️ Không đủ 4 góc để transform.")

        src_points = np.array([
            corners["top_left"],
            corners["top_right"],
            corners["bottom_left"],
            corners["bottom_right"]
        ], dtype=np.float32)

        dst_points = np.array([
            [0, 0],
            [output_size[0] - 1, 0],
            [0, output_size[1] - 1],
            [output_size[0] - 1, output_size[1] - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_image = cv2.warpPerspective(image, M, output_size)

        return transformed_image
    except Exception as e:
        print(f"❌ [ERROR] {e}")
        return None