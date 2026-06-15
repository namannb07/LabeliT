from typing import Dict, List, Optional, Tuple

from auto_annotator.parsers.base import OutputParser
from auto_annotator.parsers.detr import DetrParser
from auto_annotator.parsers.yolo import YoloParser

# Most-specific first. DetrParser checks exact tensor names;
# YoloParser matches any single-output engine, so it must come last.
PARSER_REGISTRY = [DetrParser, YoloParser]


def detect_parser(
    output_names: List[str],
    output_shapes: Dict[str, Tuple],
    num_classes_hint: Optional[int] = None,
) -> OutputParser:
    for cls in PARSER_REGISTRY:
        if cls.can_parse(output_names, output_shapes):
            return cls(output_shapes, num_classes_hint=num_classes_hint)
    raise ValueError(
        f"No parser matched engine outputs.\n"
        f"Tensors: {output_names}\n"
        f"Shapes: {output_shapes}\n"
        f"Registered parsers: {[c.__name__ for c in PARSER_REGISTRY]}"
    )


__all__ = ["OutputParser", "DetrParser", "YoloParser", "detect_parser",
           "PARSER_REGISTRY"]
