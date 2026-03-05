import numpy as np
from typing import Tuple, Optional

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def crop_bbox(image: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    """
    Safely crop an image using an xyxy bounding box.

    Args:
        image: np.ndarray (H, W, 3)
        bbox: (x1, y1, x2, y2) in pixel coordinates

    Returns:
        Cropped image as np.ndarray or None if invalid.
    """

    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")

    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Clamp bbox to image boundaries
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    # Validate bbox
    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]