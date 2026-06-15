"""Auto Annotator — TensorRT-accelerated bulk image annotation GUI for Jetson."""

__version__ = "0.3.0"

from auto_annotator.app import AnnotatorApp
from auto_annotator.detector import TRTDetector
from auto_annotator.engine_builder import OnnxModelInfo, build_engine, inspect_onnx
from auto_annotator.models import BoundingBox, ModelType
from auto_annotator.parsers import OutputParser
from auto_annotator.store import AnnotationStore

__all__ = [
    "BoundingBox",
    "ModelType",
    "OnnxModelInfo",
    "OutputParser",
    "TRTDetector",
    "AnnotationStore",
    "AnnotatorApp",
    "build_engine",
    "inspect_onnx",
    "__version__",
]
