"""LabeliT — ONNX Runtime-based bulk image annotation GUI."""

__version__ = "0.4.0"

from auto_annotator.app import AnnotatorApp
from auto_annotator.detector import OnnxDetector
from auto_annotator.models import BoundingBox, ModelType
from auto_annotator.parsers import OutputParser
from auto_annotator.store import AnnotationStore

__all__ = [
    "BoundingBox",
    "ModelType",
    "OutputParser",
    "OnnxDetector",
    "AnnotationStore",
    "AnnotatorApp",
    "__version__",
]
