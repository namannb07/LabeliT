from typing import Dict, List, Optional, Tuple

import numpy as np

from auto_annotator.models import BoundingBox, OutputBuffer
from auto_annotator.parsers.base import OutputParser, letterbox_preprocess, nms


class DetrParser(OutputParser):
    """Parser for RF-DETR (and compatible transformer detectors).

    Matches engines with two output tensors named ``"dets"`` and ``"labels"``.
    Derives num_classes from the ``"labels"`` tensor's last dimension so the
    parser adapts to any class count the engine was trained on.
    """

    def __init__(
        self,
        output_shapes: Dict[str, Tuple],
        num_classes_hint: Optional[int] = None,
    ) -> None:
        labels_shape = output_shapes["labels"]
        self._num_classes = int(labels_shape[-1])

    @staticmethod
    def can_parse(
        output_names: List[str],
        output_shapes: Dict[str, Tuple],
    ) -> bool:
        return set(output_names) == {"dets", "labels"}

    @property
    def name(self) -> str:
        return "rfdetr"

    @property
    def num_model_classes(self) -> int:
        return self._num_classes

    def preprocess(
        self, img_bgr: np.ndarray, net_w: int, net_h: int
    ) -> Tuple[np.ndarray, dict]:
        tensor, scale, pad_x, pad_y = letterbox_preprocess(img_bgr, net_w, net_h)
        return tensor, {
            "scale": scale, "pad_x": pad_x, "pad_y": pad_y,
            "net_w": net_w, "net_h": net_h,
        }

    def parse(
        self,
        output_bufs: List[OutputBuffer],
        meta: dict,
        orig_w: int,
        orig_h: int,
        conf_thresh: float,
    ) -> List[BoundingBox]:
        dets_buf = next(b for b in output_bufs if b.name == "dets")
        lbl_buf = next(b for b in output_bufs if b.name == "labels")
        dets_raw = dets_buf.host.reshape(dets_buf.shape).astype(np.float32)
        labels_raw = lbl_buf.host.reshape(lbl_buf.shape).astype(np.float32)

        if dets_raw.ndim == 3:
            dets_raw = dets_raw.squeeze(0)
        if labels_raw.ndim == 3:
            labels_raw = labels_raw.squeeze(0)

        num_queries = dets_raw.shape[0]
        nc = self._num_classes

        class_logits = labels_raw[:, :nc]
        class_ids = class_logits.argmax(axis=1)
        best_logits = class_logits[np.arange(num_queries), class_ids]
        scores = 1.0 / (1.0 + np.exp(-best_logits.astype(np.float64)))

        mask = scores >= conf_thresh
        if not mask.any():
            return []

        dets_f = dets_raw[mask]
        scores_f = scores[mask].astype(np.float32)
        class_ids_f = class_ids[mask]

        scale = meta["scale"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]
        net_w = meta["net_w"]
        net_h = meta["net_h"]

        cx_pad = dets_f[:, 0] * net_w
        cy_pad = dets_f[:, 1] * net_h
        bw_pad = dets_f[:, 2] * net_w
        bh_pad = dets_f[:, 3] * net_h

        cx_img = (cx_pad - pad_x) / scale
        cy_img = (cy_pad - pad_y) / scale
        w_img = bw_pad / scale
        h_img = bh_pad / scale

        x1 = np.clip(cx_img - w_img / 2, 0, orig_w)
        y1 = np.clip(cy_img - h_img / 2, 0, orig_h)
        x2 = np.clip(cx_img + w_img / 2, 0, orig_w)
        y2 = np.clip(cy_img + h_img / 2, 0, orig_h)

        keep = nms(np.stack([x1, y1, x2, y2], axis=1), scores_f)

        results: List[BoundingBox] = []
        for i in keep:
            w_px = x2[i] - x1[i]
            h_px = y2[i] - y1[i]
            if w_px < 1 or h_px < 1:
                continue
            cid = int(np.clip(int(class_ids_f[i]), 0, nc - 1))
            cx_n = (x1[i] + w_px / 2) / orig_w
            cy_n = (y1[i] + h_px / 2) / orig_h
            results.append(
                BoundingBox(cid, float(cx_n), float(cy_n),
                            float(w_px / orig_w), float(h_px / orig_h),
                            float(scores_f[i]))
            )
        return results
