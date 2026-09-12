from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from auto_annotator.models import BoundingBox, OutputBuffer
from auto_annotator.parsers import OutputParser, detect_parser

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class OnnxDetector:
    def __init__(self):
        self.model_path: Optional[Path] = None
        self.labels: List[str] = []
        self.num_model_classes: int = 0
        self.net_w: int = 0
        self.net_h: int = 0
        self._session: Optional["ort.InferenceSession"] = None
        self._input_name: Optional[str] = None
        self._output_names: List[str] = []
        self._output_shapes: Dict[str, Tuple] = {}
        self._parser: Optional[OutputParser] = None

    @property
    def model_type(self) -> Optional[str]:
        return self._parser.name if self._parser else None

    def load(self, model_path: Path, labels: List[str]) -> None:
        if not ORT_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not installed.\n"
                "Install it with:  pip install onnxruntime"
            )
        try:
            self._load_impl(model_path, labels)
        except Exception as e:
            raise RuntimeError(f"Model load failed: {e}") from e

    def _load_impl(self, model_path: Path, labels: List[str]) -> None:
        self.model_path = model_path
        self.labels = labels

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Inspect input tensor
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no input tensors.")

        inp = inputs[0]
        self._input_name = inp.name
        shape = inp.shape

        if len(shape) < 4:
            raise RuntimeError(
                f"Unexpected input tensor shape {tuple(shape)} — "
                "expected (batch, channels, H, W)"
            )

        h, w = shape[2], shape[3]
        # Handle dynamic dimensions (e.g. -1 or string like 'height')
        if not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0:
            raise RuntimeError(
                f"Model has dynamic spatial dimensions (H={h}, W={w}). "
                "Only fixed-resolution models are supported."
            )
        self.net_h = h
        self.net_w = w

        # Collect output tensor info
        self._output_names = []
        self._output_shapes = {}
        for out in self._session.get_outputs():
            self._output_names.append(out.name)
            # Replace any dynamic dims with 1 for shape detection
            resolved_shape = tuple(
                d if isinstance(d, int) and d > 0 else 1
                for d in out.shape
            )
            self._output_shapes[out.name] = resolved_shape

        self._parser = detect_parser(
            self._output_names,
            self._output_shapes,
            num_classes_hint=len(labels),
        )
        self.num_model_classes = self._parser.num_model_classes

        if len(self.labels) < self.num_model_classes:
            for i in range(len(self.labels), self.num_model_classes):
                self.labels.append(f"class_{i}")
        elif len(self.labels) > self.num_model_classes:
            self.labels = self.labels[: self.num_model_classes]

    def run(self, image_path: Path, conf_thresh: float) -> List[BoundingBox]:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return []
        orig_h, orig_w = img_bgr.shape[:2]

        tensor, meta = self._parser.preprocess(img_bgr, self.net_w, self.net_h)

        inp = tensor.astype(np.float32)
        raw_outputs = self._session.run(self._output_names, {self._input_name: inp})

        output_bufs: List[OutputBuffer] = []
        for name, raw in zip(self._output_names, raw_outputs):
            arr = np.asarray(raw, dtype=np.float32)
            output_bufs.append(
                OutputBuffer(
                    name=name,
                    host=arr.ravel(),
                    shape=arr.shape,
                    dtype=arr.dtype,
                )
            )

        return self._parser.parse(output_bufs, meta, orig_w, orig_h, conf_thresh)

    def close(self) -> None:
        self._session = None
