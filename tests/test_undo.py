from pathlib import Path

from auto_annotator.models import BoundingBox
from auto_annotator.ui.undo import UndoStack


def _box(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, conf=0.9):
    return BoundingBox(class_id, cx, cy, w, h, conf)


def test_push_then_pop_returns_pushed_snapshot():
    stack = UndoStack()
    p = Path("a.jpg")
    stack.push(p, [_box(class_id=0)])
    assert stack.can_undo(p)
    restored = stack.pop(p)
    assert restored is not None
    assert len(restored) == 1
    assert restored[0].class_id == 0


def test_pop_on_empty_returns_none():
    assert UndoStack().pop(Path("a.jpg")) is None


def test_push_deep_copies_boxes():
    stack = UndoStack()
    p = Path("a.jpg")
    boxes = [_box(class_id=1)]
    stack.push(p, boxes)
    boxes[0].class_id = 99
    restored = stack.pop(p)
    assert restored[0].class_id == 1


def test_history_is_per_path():
    stack = UndoStack()
    a, b = Path("a.jpg"), Path("b.jpg")
    stack.push(a, [_box(0)])
    assert stack.can_undo(a)
    assert not stack.can_undo(b)


def test_depth_cap_drops_oldest():
    stack = UndoStack(depth_max=3)
    p = Path("a.jpg")
    for i in range(5):
        stack.push(p, [_box(class_id=i)])
    assert stack.pop(p)[0].class_id == 4
    assert stack.pop(p)[0].class_id == 3
    assert stack.pop(p)[0].class_id == 2
    assert stack.pop(p) is None


def test_discard_last_drops_without_returning():
    stack = UndoStack()
    p = Path("a.jpg")
    stack.push(p, [_box(class_id=0)])
    stack.push(p, [_box(class_id=1)])
    stack.discard_last(p)
    assert stack.pop(p)[0].class_id == 0


def test_discard_last_on_empty_is_safe():
    UndoStack().discard_last(Path("a.jpg"))


def test_push_with_none_path_is_noop():
    stack = UndoStack()
    stack.push(None, [_box(0)])
    assert not stack.can_undo(None)


def test_clear_specific_path():
    stack = UndoStack()
    a, b = Path("a.jpg"), Path("b.jpg")
    stack.push(a, [_box(0)])
    stack.push(b, [_box(1)])
    stack.clear(a)
    assert not stack.can_undo(a)
    assert stack.can_undo(b)


def test_clear_all():
    stack = UndoStack()
    stack.push(Path("a.jpg"), [_box(0)])
    stack.push(Path("b.jpg"), [_box(1)])
    stack.clear()
    assert not stack.can_undo(Path("a.jpg"))
    assert not stack.can_undo(Path("b.jpg"))
