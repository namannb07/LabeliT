import numpy as np

from auto_annotator.parsers.base import letterbox_preprocess, nms


def test_letterbox_square_to_square_has_no_padding():
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    tensor, scale, pad_x, pad_y = letterbox_preprocess(img, 640, 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert scale == 1.0
    assert pad_x == 0
    assert pad_y == 0


def test_letterbox_wide_image_pads_vertically():
    img = np.zeros((100, 640, 3), dtype=np.uint8)
    _, scale, pad_x, pad_y = letterbox_preprocess(img, 640, 640)
    assert scale == 1.0
    assert pad_x == 0
    assert pad_y > 0


def test_letterbox_tall_image_pads_horizontally():
    img = np.zeros((640, 100, 3), dtype=np.uint8)
    _, scale, pad_x, pad_y = letterbox_preprocess(img, 640, 640)
    assert scale == 1.0
    assert pad_y == 0
    assert pad_x > 0


def test_letterbox_downscales_oversized_image():
    img = np.zeros((1280, 1280, 3), dtype=np.uint8)
    tensor, scale, _, _ = letterbox_preprocess(img, 640, 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert scale == 0.5


def test_letterbox_output_is_float32_chw_with_batch_dim():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, _, _, _ = letterbox_preprocess(img, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3


def test_nms_drops_overlapping_lower_score_boxes():
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [10, 10, 110, 110],
            [500, 500, 600, 600],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_thresh=0.5)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep


def test_nms_keeps_disjoint_boxes():
    boxes = np.array(
        [
            [0, 0, 50, 50],
            [100, 100, 150, 150],
            [200, 200, 250, 250],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_thresh=0.5)
    assert set(keep) == {0, 1, 2}


def test_nms_single_box_kept():
    boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    assert nms(boxes, scores) == [0]
