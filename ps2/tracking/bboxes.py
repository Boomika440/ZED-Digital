"""
Bounding Box Geometry and Overlap Metrics.
Provides coordinate transformations between:
- [x1, y1, x2, y2] (top-left, bottom-right)
- [xc, yc, w, h]  (center x, center y, width, height)
- [xc, yc, a, h]  (center x, center y, aspect ratio w/h, height)
Along with Vectorized IoU, DIoU, and Inter-Track Overlap Ratio (ITOR) for occlusion detection.
"""

import numpy as np


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Converts [x1, y1, x2, y2] to [xc, yc, w, h]."""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    w = np.maximum(0.0, boxes[..., 2] - boxes[..., 0])
    h = np.maximum(0.0, boxes[..., 3] - boxes[..., 1])
    xc = boxes[..., 0] + w * 0.5
    yc = boxes[..., 1] + h * 0.5
    return np.stack([xc, yc, w, h], axis=-1)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Converts [xc, yc, w, h] to [x1, y1, x2, y2]."""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    x1 = boxes[..., 0] - boxes[..., 2] * 0.5
    y1 = boxes[..., 1] - boxes[..., 3] * 0.5
    x2 = boxes[..., 0] + boxes[..., 2] * 0.5
    y2 = boxes[..., 1] + boxes[..., 3] * 0.5
    return np.stack([x1, y1, x2, y2], axis=-1)


def xyxy_to_xyah(boxes: np.ndarray) -> np.ndarray:
    """Converts [x1, y1, x2, y2] to Kalman state measurement [xc, yc, a, h] where a = w / h."""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    w = np.maximum(1e-4, boxes[..., 2] - boxes[..., 0])
    h = np.maximum(1e-4, boxes[..., 3] - boxes[..., 1])
    xc = boxes[..., 0] + w * 0.5
    yc = boxes[..., 1] + h * 0.5
    a = w / h
    return np.stack([xc, yc, a, h], axis=-1)


def xyah_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Converts Kalman state measurement [xc, yc, a, h] to [x1, y1, x2, y2]."""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    w = boxes[..., 2] * boxes[..., 3]
    h = boxes[..., 3]
    x1 = boxes[..., 0] - w * 0.5
    y1 = boxes[..., 1] - h * 0.5
    x2 = boxes[..., 0] + w * 0.5
    y2 = boxes[..., 1] + h * 0.5
    return np.stack([x1, y1, x2, y2], axis=-1)


def box_ious(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Intersection-over-Union (IoU) matrix between boxes_a and boxes_b.
    Both inputs must be [N, 4] and [M, 4] in [x1, y1, x2, y2] format.
    Returns: [N, M] matrix with IoU in [0.0, 1.0].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])

    # Pairwise intersection coordinates
    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    iou = np.where(union_area > 0, inter_area / np.maximum(union_area, 1e-6), 0.0)
    return iou.astype(np.float32)


def compute_inter_track_overlap(boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Computes Inter-Track Overlap Ratio (ITOR) for each bounding box in a frame.
    ITOR_i = max_{j != i} Area(B_i ∩ B_j) / Area(B_i)
    
    If ITOR_i > threshold (e.g. 0.35), person i is undergoing mutual occlusion,
    indicating their appearance should NOT be updated to avoid identity drift.
    """
    n = len(boxes_xyxy)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    area = np.maximum(1e-4, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

    inter_x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    inter_y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    inter_x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    inter_y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Zero out diagonal (self-intersection)
    np.fill_diagonal(inter_area, 0.0)

    # Overlap fraction relative to each box's own area: [N, N]
    overlap_ratios = inter_area / area[:, None]
    max_overlaps = np.max(overlap_ratios, axis=1)
    return max_overlaps.astype(np.float32)
