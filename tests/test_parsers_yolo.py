import numpy as np
import pytest

from auto_annotator.models import OutputBuffer
from auto_annotator.parsers.yolo import YoloParser


def make_buffer(name: str, arr: np.ndarray) -> OutputBuffer:
    return OutputBuffer(
        name=name,
        host=arr.flatten(),
        shape=arr.shape,
        dtype=arr.dtype,
    )


def test_can_parse_accepts_single_2d_output():
    assert YoloParser.can_parse(["output0"], {"output0": (1, 84, 8400)})
    assert YoloParser.can_parse(["output0"], {"output0": (1, 8400, 84)})


def test_can_parse_rejects_two_output_engine():
    assert not YoloParser.can_parse(
        ["dets", "labels"],
        {"dets": (1, 300, 4), "labels": (1, 300, 80)},
    )


def test_num_classes_inferred_from_shape():
    parser = YoloParser(output_shapes={"output0": (1, 84, 8400)})
    assert parser.num_model_classes == 80


def test_yolo_v8_layout_decodes_one_high_confidence_box():
    nc = 3
    num_anchors = 200
    out = np.zeros((1, 4 + nc, num_anchors), dtype=np.float32)
    out[0, 0, 0] = 320
    out[0, 1, 0] = 320
    out[0, 2, 0] = 100
    out[0, 3, 0] = 100
    out[0, 4 + 1, 0] = 0.9

    parser = YoloParser(output_shapes={"output0": (1, 4 + nc, num_anchors)})
    buf = make_buffer("output0", out)
    meta = {"scale": 1.0, "pad_x": 0, "pad_y": 0}
    boxes = parser.parse([buf], meta, orig_w=640, orig_h=640, conf_thresh=0.25)

    assert len(boxes) == 1
    assert boxes[0].class_id == 1
    assert boxes[0].confidence == pytest.approx(0.9, abs=1e-5)
    assert boxes[0].cx == pytest.approx(0.5, abs=1e-3)
    assert boxes[0].cy == pytest.approx(0.5, abs=1e-3)


def test_below_conf_threshold_is_filtered_out():
    nc = 2
    out = np.zeros((1, 4 + nc, 100), dtype=np.float32)
    out[0, 0, 0] = 100
    out[0, 1, 0] = 100
    out[0, 2, 0] = 50
    out[0, 3, 0] = 50
    out[0, 4, 0] = 0.1

    parser = YoloParser(output_shapes={"output0": (1, 4 + nc, 100)})
    buf = make_buffer("output0", out)
    boxes = parser.parse(
        [buf],
        {"scale": 1.0, "pad_x": 0, "pad_y": 0},
        orig_w=640,
        orig_h=640,
        conf_thresh=0.25,
    )
    assert boxes == []


def test_v5_objectness_layout_with_hint():
    nc = 4
    num_anchors = 50
    out = np.zeros((1, 5 + nc, num_anchors), dtype=np.float32)
    out[0, 0, 0] = 100
    out[0, 1, 0] = 100
    out[0, 2, 0] = 40
    out[0, 3, 0] = 40
    out[0, 4, 0] = 0.95
    out[0, 5 + 2, 0] = 0.9

    parser = YoloParser(
        output_shapes={"output0": (1, 5 + nc, num_anchors)},
        num_classes_hint=nc,
    )
    assert parser.name == "yolo5"
    buf = make_buffer("output0", out)
    boxes = parser.parse(
        [buf],
        {"scale": 1.0, "pad_x": 0, "pad_y": 0},
        orig_w=640,
        orig_h=640,
        conf_thresh=0.25,
    )
    assert len(boxes) == 1
    assert boxes[0].class_id == 2
