from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import cv2
import numpy as np

from auto_annotator.config import NMS_IOU
from auto_annotator.models import BoundingBox, TRTBuffer


class OutputParser(ABC):

    @staticmethod
    @abstractmethod
    def can_parse(
        output_names: List[str],
        output_shapes: Dict[str, Tuple],
    ) -> bool:
        """Return True if this parser handles the given engine output configuration."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical identifier for display (e.g. 'yolo11', 'rfdetr')."""

    @property
    @abstractmethod
    def num_model_classes(self) -> int:
        """Number of classes the model was trained on, derived from engine output shapes."""

    @abstractmethod
    def preprocess(
        self, img_bgr: np.ndarray, net_w: int, net_h: int
    ) -> Tuple[np.ndarray, dict]:
        """Prepare image for inference. Returns (tensor, metadata_dict).

        The metadata dict is opaque — each parser stores whatever its
        ``parse`` method needs for coordinate conversion.
        """

    @abstractmethod
    def parse(
        self,
        output_bufs: List[TRTBuffer],
        meta: dict,
        orig_w: int,
        orig_h: int,
        conf_thresh: float,
    ) -> List[BoundingBox]:
        """Decode output buffers into bounding boxes."""


def letterbox_preprocess(
    img_bgr: np.ndarray, net_w: int, net_h: int
) -> Tuple[np.ndarray, float, int, int]:
    """Letterbox resize with aspect-ratio preservation.

    Returns (tensor, scale, pad_x, pad_y).
    """
    orig_h, orig_w_px = img_bgr.shape[:2]
    scale = min(net_w / orig_w_px, net_h / orig_h)
    new_w = int(orig_w_px * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (net_w - new_w) // 2
    pad_y = (net_h - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_y, net_h - new_h - pad_y,
        pad_x, net_w - new_w - pad_x,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
    return tensor, scale, pad_x, pad_y


def nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = NMS_IOU,
) -> List[int]:
    """Greedy IoU-based non-maximum suppression."""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = (
        boxes_xyxy[:, 0], boxes_xyxy[:, 1],
        boxes_xyxy[:, 2], boxes_xyxy[:, 3],
    )
    areas = (x2 - x1) * (y2 - y1)
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return keep
