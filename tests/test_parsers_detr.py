import numpy as np
import pytest

from auto_annotator.models import TRTBuffer
from auto_annotator.parsers.detr import DetrParser


def make_buffer(name: str, arr: np.ndarray) -> TRTBuffer:
    return TRTBuffer(
        name=name,
        host=arr.flatten(),
        device=None,
        shape=arr.shape,
        dtype=arr.dtype,
    )


def test_can_parse_matches_named_two_output_engine():
    shapes = {"dets": (1, 300, 4), "labels": (1, 300, 80)}
    assert DetrParser.can_parse(["dets", "labels"], shapes)
    assert DetrParser.can_parse(["labels", "dets"], shapes)


def test_can_parse_rejects_single_output_yolo_style():
    assert not DetrParser.can_parse(
        ["output0"], {"output0": (1, 84, 8400)}
    )


def test_num_classes_from_labels_tensor():
    parser = DetrParser(
        output_shapes={"dets": (1, 300, 4), "labels": (1, 300, 10)}
    )
    assert parser.num_model_classes == 10


def test_decodes_one_high_score_query():
    nc = 5
    num_q = 100
    net_w = 560
    net_h = 560

    dets = np.zeros((1, num_q, 4), dtype=np.float32)
    labels = np.full((1, num_q, nc), -10.0, dtype=np.float32)
    dets[0, 0] = [0.5, 0.5, 0.4, 0.4]
    labels[0, 0, 2] = 4.0

    parser = DetrParser(
        output_shapes={"dets": (1, num_q, 4), "labels": (1, num_q, nc)}
    )
    bufs = [make_buffer("dets", dets), make_buffer("labels", labels)]
    meta = {
        "scale": 1.0,
        "pad_x": 0,
        "pad_y": 0,
        "net_w": net_w,
        "net_h": net_h,
    }
    boxes = parser.parse(bufs, meta, orig_w=net_w, orig_h=net_h, conf_thresh=0.25)

    assert len(boxes) == 1
    assert boxes[0].class_id == 2
    assert boxes[0].confidence > 0.9
    assert boxes[0].cx == pytest.approx(0.5, abs=1e-3)
    assert boxes[0].w == pytest.approx(0.4, abs=1e-3)


def test_low_logits_return_no_boxes():
    nc = 5
    num_q = 50
    dets = np.zeros((1, num_q, 4), dtype=np.float32)
    labels = np.full((1, num_q, nc), -10.0, dtype=np.float32)
    dets[0, 0] = [0.5, 0.5, 0.4, 0.4]

    parser = DetrParser(
        output_shapes={"dets": (1, num_q, 4), "labels": (1, num_q, nc)}
    )
    bufs = [make_buffer("dets", dets), make_buffer("labels", labels)]
    meta = {"scale": 1.0, "pad_x": 0, "pad_y": 0, "net_w": 560, "net_h": 560}
    boxes = parser.parse(bufs, meta, orig_w=560, orig_h=560, conf_thresh=0.25)
    assert boxes == []
