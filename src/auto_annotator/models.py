from dataclasses import dataclass
from typing import Tuple

import numpy as np


class ModelType:
    YOLO11 = "yolo11"
    RFDETR = "rfdetr"


@dataclass
class OutputBuffer:
    name: str
    host: np.ndarray
    shape: Tuple
    dtype: np.dtype


@dataclass
class BoundingBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float
