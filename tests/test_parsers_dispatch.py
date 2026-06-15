import pytest

from auto_annotator.parsers import detect_parser
from auto_annotator.parsers.detr import DetrParser
from auto_annotator.parsers.yolo import YoloParser


def test_detect_yolo_for_single_output_engine():
    parser = detect_parser(["output0"], {"output0": (1, 84, 8400)})
    assert isinstance(parser, YoloParser)


def test_detect_detr_for_dets_and_labels_engine():
    parser = detect_parser(
        ["dets", "labels"],
        {"dets": (1, 300, 4), "labels": (1, 300, 91)},
    )
    assert isinstance(parser, DetrParser)


def test_detect_raises_for_unknown_engine_shape():
    with pytest.raises(ValueError):
        detect_parser(["foo", "bar"], {"foo": (1, 1), "bar": (1, 1)})
