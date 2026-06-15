from typing import Dict, List, Optional, Tuple

import numpy as np

from auto_annotator.models import BoundingBox, OutputBuffer
from auto_annotator.parsers.base import OutputParser, letterbox_preprocess, nms


class YoloParser(OutputParser):
    """Parser for the YOLO family (v5/v7/v8/v9/v10/v11).

    Derives num_classes and layout from the engine output shape directly.
    Uses an optional ``num_classes_hint`` (from the labels file) only to
    disambiguate YOLOv5 (objectness column) from v8+ when shapes are
    ambiguous.
    """

    def __init__(
        self,
        output_shapes: Dict[str, Tuple],
        num_classes_hint: Optional[int] = None,
    ) -> None:
        output_name = next(iter(output_shapes))
        shape = output_shapes[output_name]
        dims = [int(d) for d in shape if int(d) != 1]

        self._output_name = output_name

        if dims[0] < dims[1]:
            feat_dim, det_count = dims[0], dims[1]
        else:
            feat_dim, det_count = dims[1], dims[0]
        self._needs_transpose = dims[0] == feat_dim

        if num_classes_hint is not None and feat_dim == 5 + num_classes_hint:
            self._has_objectness = True
            self._num_classes = num_classes_hint
        elif num_classes_hint is not None and feat_dim == 4 + num_classes_hint:
            self._has_objectness = False
            self._num_classes = num_classes_hint
        else:
            self._has_objectness = False
            self._num_classes = feat_dim - 4

    @staticmethod
    def can_parse(
        output_names: List[str],
        output_shapes: Dict[str, Tuple],
    ) -> bool:
        if len(output_names) != 1:
            return False
        shape = output_shapes[output_names[0]]
        dims = [int(d) for d in shape if int(d) != 1]
        if len(dims) != 2:
            return False
        return min(dims) >= 5

    @property
    def name(self) -> str:
        return "yolo5" if self._has_objectness else "yolo11"

    @property
    def num_model_classes(self) -> int:
        return self._num_classes

    def preprocess(
        self, img_bgr: np.ndarray, net_w: int, net_h: int
    ) -> Tuple[np.ndarray, dict]:
        tensor, scale, pad_x, pad_y = letterbox_preprocess(img_bgr, net_w, net_h)
        return tensor, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y}

    def parse(
        self,
        output_bufs: List[OutputBuffer],
        meta: dict,
        orig_w: int,
        orig_h: int,
        conf_thresh: float,
    ) -> List[BoundingBox]:
        buf = next(b for b in output_bufs if b.name == self._output_name)
        raw = buf.host.reshape(buf.shape)
        out = raw.squeeze(0).astype(np.float32)

        if self._needs_transpose:
            preds = out.T
        else:
            preds = out

        boxes_cxywh = preds[:, :4]
        nc = self._num_classes

        if self._has_objectness:
            objectness = preds[:, 4]
            class_scores = preds[:, 5 : 5 + nc]
            scores = objectness * class_scores.max(axis=1)
        else:
            class_scores = preds[:, 4 : 4 + nc]
            scores = class_scores.max(axis=1)

        class_ids = class_scores.argmax(axis=1)

        mask = scores >= conf_thresh
        if not mask.any():
            return []

        boxes_cxywh = boxes_cxywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        scale = meta["scale"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]

        cx_img = (boxes_cxywh[:, 0] - pad_x) / scale
        cy_img = (boxes_cxywh[:, 1] - pad_y) / scale
        w_img = boxes_cxywh[:, 2] / scale
        h_img = boxes_cxywh[:, 3] / scale

        x1 = np.clip(cx_img - w_img / 2, 0, orig_w)
        y1 = np.clip(cy_img - h_img / 2, 0, orig_h)
        x2 = np.clip(cx_img + w_img / 2, 0, orig_w)
        y2 = np.clip(cy_img + h_img / 2, 0, orig_h)

        keep = nms(np.stack([x1, y1, x2, y2], axis=1), scores)

        results: List[BoundingBox] = []
        for i in keep:
            bw = x2[i] - x1[i]
            bh = y2[i] - y1[i]
            if bw < 1 or bh < 1:
                continue
            cid = int(np.clip(int(class_ids[i]), 0, nc - 1))
            cx_n = (x1[i] + bw / 2) / orig_w
            cy_n = (y1[i] + bh / 2) / orig_h
            results.append(
                BoundingBox(cid, float(cx_n), float(cy_n),
                            float(bw / orig_w), float(bh / orig_h),
                            float(scores[i]))
            )
        return results
