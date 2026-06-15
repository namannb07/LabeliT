from dataclasses import dataclass

from auto_annotator.models import BoundingBox


@dataclass
class BoxView:
    """UI overlay wrapping a domain BoundingBox with transient render state.

    Never persisted, never exported. Created by AnnotatorApp when rendering
    and discarded when boxes are passed to the store.
    """

    box: BoundingBox
    selected: bool = False
    canvas_x1: float = 0.0
    canvas_y1: float = 0.0
    canvas_x2: float = 0.0
    canvas_y2: float = 0.0
