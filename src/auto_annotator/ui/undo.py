from pathlib import Path
from typing import Dict, List, Optional

from auto_annotator.config import UNDO_DEPTH_MAX
from auto_annotator.models import BoundingBox


class UndoStack:
    """Per-image undo history. Each push stores a snapshot copy of the box list."""

    def __init__(self, depth_max: int = UNDO_DEPTH_MAX):
        self._depth_max = depth_max
        self._history: Dict[Path, List[List[BoundingBox]]] = {}

    def push(self, path: Optional[Path], boxes: List[BoundingBox]) -> None:
        if path is None:
            return
        hist = self._history.setdefault(path, [])
        hist.append([
            BoundingBox(b.class_id, b.cx, b.cy, b.w, b.h, b.confidence)
            for b in boxes
        ])
        if len(hist) > self._depth_max:
            hist.pop(0)

    def pop(self, path: Optional[Path]) -> Optional[List[BoundingBox]]:
        if path is None:
            return None
        hist = self._history.get(path)
        if not hist:
            return None
        return hist.pop()

    def discard_last(self, path: Optional[Path]) -> None:
        # Undo a speculative push() when the action it guarded turned out to be
        # a no-op (e.g. a click in draw-mode without a meaningful drag).
        if path is None:
            return
        hist = self._history.get(path)
        if hist:
            hist.pop()

    def can_undo(self, path: Optional[Path]) -> bool:
        if path is None:
            return False
        return bool(self._history.get(path))

    def clear(self, path: Optional[Path] = None) -> None:
        if path is None:
            self._history.clear()
        else:
            self._history.pop(path, None)
