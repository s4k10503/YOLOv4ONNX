"""
Quoted and modified from:
https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/dependencies/inference.ipynb

MIT License
Copyright (c) 2020 Jennifer Wang
"""

import numpy as np
import cv2


def image_preprocess(image: np.ndarray, target_size: tuple, gt_boxes: np.ndarray = None) -> np.ndarray:  # type: ignore
    """
    Preprocess the image by resizing and padding.

    Args:
        image (np.ndarray): Input image array of shape (height, width, channels).
        target_size (tuple): Target size for resizing (height, width).
        gt_boxes (np.ndarray, optional): Ground truth boxes. Defaults to None.

    Returns:
        np.ndarray: Resized and padded image.
    """

    ih, iw = target_size
    h, w, _ = image.shape

    # Calculate the scale and new width/height
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)

    # Resize the image
    image_resized = cv2.resize(image, (nw, nh))

    # Create a padded image with the new dimensions
    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_padded = image_padded / 255.

    # If ground truth boxes are provided, adjust their coordinates
    if gt_boxes is not None:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes

    return image_padded
