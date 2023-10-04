"""
Quoted and modified from:
https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/dependencies/inference.ipynb

MIT License
Copyright (c) 2020 Jennifer Wang
"""

import time
import cv2
import numpy as np
from typing import Any, Union

from utils.preprocessing import image_preprocess
from utils.postprocessing import postprocess_bbbox, postprocess_boxes, nms
from utils.visualization import draw_bbox


def get_anchors(anchors_path: str, tiny: bool = False) -> np.ndarray:
    """
    Read anchor values from a file.

    Args:
        anchors_path (str): Path to the file containing anchor values.
        tiny (bool, optional): Flag to indicate if tiny anchors are used. Defaults to False.

    Returns:
        np.ndarray: Anchors reshaped to (3, 3, 2).
    """

    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)

    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)


def detect_objects(sess: Any, image: np.ndarray, anchors: np.ndarray, classes: dict, input_size: int = 416, tiny: bool = False) -> np.ndarray:
    """
    Detect objects in an image using a trained ONNX model.

    Args:
        sess (Any): ONNX runtime session containing the model.
        image (np.ndarray): Input image array of shape (height, width, channels).
        anchors (np.ndarray): Anchor values for detection.
        classes (dict): Dictionary containing class names.
        input_size (int, optional): Size of the input image after preprocessing. Defaults to 416.
        tiny (bool, optional): If using YOLOv4 Tiny model. Defaults to False.

    Returns:
        np.ndarray: Image with detected objects and bounding boxes drawn.
    """

    # Preprocess image and prepare for inference
    original_image_size = image.shape[:2]
    image_data = image_preprocess(np.copy(image), [input_size, input_size])
    # (height, width, channels) --> (1, height, width, channels)
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    # Run inference using ONNX runtime
    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name
    detections = sess.run(output_names, {input_name: image_data})

    # Post-process detections and draw bounding boxes

    if tiny:
        STRIDES = np.array([16, 32])
        XYSCALE = [1.05, 1.05]
    else:
        STRIDES = np.array([8, 16, 32])
        XYSCALE = [1.2, 1.1, 1.05]

    pred_bbox = postprocess_bbbox(detections, anchors, STRIDES, XYSCALE)
    bboxes = postprocess_boxes(
        pred_bbox, original_image_size, input_size, 0.25)
    bboxes = nms(bboxes, 0.213, method='nms')

    return draw_bbox(image, bboxes, classes)


def process_video(sess: Any, anchors: np.ndarray, input_source: Union[str, int], classes: dict) -> None:
    """
    Process a video for object detection and display detected objects.

    Args:
        sess (Any): ONNX runtime session containing the model.
        anchors (np.ndarray): Array of anchor values for detection.
        input_source (Union[str, int]): Path to the video file or device index.
        classes (dict): Dictionary of class names.

    Returns:
        None
    """

    cap = cv2.VideoCapture(input_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            detected_image = detect_objects(sess, frame, anchors, classes)
            end_time = time.time()

            inference_time = (end_time - start_time) * 1000
            inference_time_text = f"Inference time: {inference_time:.2f} ms"

            cv2.putText(detected_image, inference_time_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Detected Objects', detected_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()


def process_image(sess: Any, anchors: np.ndarray, input_source: str, classes: dict) -> None:
    """
    Process an image for object detection and display detected objects.

    Args:
        sess (Any): ONNX runtime session containing the model.
        anchors (np.ndarray): Array of anchor values for detection.
        input_source (str): Path to the image file.
        classes (dict): Dictionary of class names.

    Returns:
        None
    """

    image = cv2.imread(input_source)

    start_time = time.time()
    detected_image = detect_objects(sess, image, anchors, classes)
    end_time = time.time()

    inference_time = (end_time - start_time) * 1000
    inference_time_text = f"Inference time: {inference_time:.2f} ms"

    cv2.putText(detected_image, inference_time_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Detected Objects', detected_image)
    cv2.waitKey(0)
