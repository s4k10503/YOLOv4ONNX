"""
Quoted and modified from:
https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/dependencies/inference.ipynb

MIT License
Copyright (c) 2020 Jennifer Wang
"""

import numpy as np
from scipy import special


def postprocess_bbbox(pred_bbox: np.ndarray, ANCHORS: np.ndarray, STRIDES: np.ndarray, XYSCALE: list[float] = [1, 1, 1]) -> np.ndarray:
    """
    Postprocess bounding box predictions to get final predictions.

    Args:
        pred_bbox (np.ndarray): Predicted bounding boxes.
        ANCHORS (np.ndarray): Anchor values for bounding boxes.
        STRIDES (np.ndarray): Stride values for bounding boxes.
        XYSCALE (list[float], optional): Scaling factors for bounding boxes. Defaults to [1, 1, 1].

    Returns:
        np.ndarray: Postprocessed bounding boxes.
    """

    for i, pred in enumerate(pred_bbox):
        conv_shape = pred.shape
        output_size = conv_shape[1]

        # Extract dx, dy, dw, dh
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]

        # Generate the grid
        xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
        xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)
        xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(np.float32)

        # Calculate pred_xy and pred_wh
        pred_xy = ((special.expit(conv_raw_dxdy) *
                   XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i])
        pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

    # Reshape and concatenate the bounding boxes
    pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = np.concatenate(pred_bbox, axis=0)
    return pred_bbox


def postprocess_boxes(pred_bbox: np.ndarray, org_img_shape: tuple, input_size: int, score_threshold: float) -> np.ndarray:
    """
    Postprocess bounding boxes by resizing and removing invalid ones.

    Args:
        pred_bbox (np.ndarray): Predicted bounding boxes.
        org_img_shape (tuple): Original image shape (height, width).
        input_size (int): Size of the input image after preprocessing.
        score_threshold (float): Threshold for valid bounding boxes.

    Returns:
        np.ndarray: Resized and filtered bounding boxes.
    """

    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)
    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    # (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)
    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    # Resize coordinates
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # clip some boxes that are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or(
        (pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(
        pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and(
        (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def bboxes_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (np.ndarray): An array of shape (N, 4) representing N bounding boxes.
                            Each bounding box is represented by [xmin, ymin, xmax, ymax].
        boxes2 (np.ndarray): An array of shape (M, 4) representing M bounding boxes.
                            Each bounding box is represented by [xmin, ymin, xmax, ymax].

    Returns:
        np.ndarray: An array of shape (N, M) representing the IoU scores between each pair of bounding boxes
                    from `boxes1` and `boxes2`.
    """

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    # Calculate area for both bounding boxes
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
        (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
        (boxes2[..., 3] - boxes2[..., 1])

    # Find the intersection coordinates
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # Calculate intersection area
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate IOU
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes: np.ndarray, iou_threshold: float, sigma: float = 0.3, method: str = 'soft-nms') -> list[np.ndarray]:
    """
    Perform Non-Max Suppression (NMS) on bounding boxes.

    Args:
        bboxes (np.ndarray): Bounding boxes in the format (xmin, ymin, xmax, ymax, score, class).
        iou_threshold (float): Threshold for IOU (Intersection Over Union).
        sigma (float, optional): Parameter for soft-NMS. Defaults to 0.3.
        method (str, optional): Method for NMS, either 'nms' or 'soft-nms'. Defaults to 'nms'.

    Returns:
        list[np.ndarray]: List of best bounding boxes after NMS.
    """

    # Extract unique classes
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask].astype(np.float32)

        # Continue until no more bounding boxes for the class
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes
