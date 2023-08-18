"""
Quoted and modified from:
https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/dependencies/inference.ipynb

MIT License
Copyright (c) 2020 Jennifer Wang
"""

import random
import colorsys
import numpy as np
import cv2


def draw_bbox(image: np.ndarray, bboxes: np.ndarray, classes: dict, show_label: bool = True) -> np.ndarray:
    """
    Draw bounding boxes on the image with color coding and labels.

    Args:
        image (np.ndarray): Input image array of shape (height, width, channels).
        bboxes (np.ndarray): Array of bounding boxes (x_min, y_min, x_max, y_max, score, class).
        classes (dict): Dictionary of class names.
        show_label (bool, optional): Flag to indicate if labels should be shown. Defaults to True.

    Returns:
        np.ndarray: Image with bounding boxes drawn.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape

    # Create unique colors for each class
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    # Draw bounding boxes for each detected object
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 300)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            # Add label with class name and confidence score
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(
                bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(
                image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                        (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image
