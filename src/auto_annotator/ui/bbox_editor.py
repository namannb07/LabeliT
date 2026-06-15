import tkinter as tk
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from auto_annotator.config import HANDLE_SIZE, MIN_BOX_DRAW_PX, MIN_BOX_RESIZE_PX
from auto_annotator.models import BoundingBox
from auto_annotator.ui.box_view import BoxView
from auto_annotator.ui.canvas_transform import CanvasTransform
from auto_annotator.ui.undo import UndoStack


class BboxEditor:
    """Mouse + keyboard state machine for drawing, selecting, moving, and
    resizing bounding boxes on a tk.Canvas.

    The editor owns its own tool / drag / draw state but does not own the
    box list - the app does. Accessors let the editor read the live state;
    callbacks let the app react to mutations.

    on_changed(committed=True)  -> persist + redraw
    on_changed(committed=False) -> redraw only (mid-drag visual update)
    on_selection_changed(class_id_or_None) -> app refreshes class-button highlight
    """

    def __init__(
        self,
        canvas: tk.Canvas,
        transform: CanvasTransform,
        undo_stack: UndoStack,
        get_current_path: Callable[[], Optional[Path]],
        get_current_boxes: Callable[[], List[BoxView]],
        get_class_enabled: Callable[[int], bool],
        get_draw_class: Callable[[], int],
        on_changed: Callable[[bool], None],
        on_selection_changed: Callable[[Optional[int]], None],
    ):
        self.canvas = canvas
        self.transform = transform
        self.undo_stack = undo_stack
        self._get_current_path = get_current_path
        self._get_current_boxes = get_current_boxes
        self._get_class_enabled = get_class_enabled
        self._get_draw_class = get_draw_class
        self._on_changed = on_changed
        self._on_selection_changed = on_selection_changed

        self.tool_mode: str = "select"
        self._draw_start: Optional[Tuple[float, float]] = None
        self._draw_preview_id = None
        self._drag_mode: Optional[str] = None
        self._drag_start_canvas: Optional[Tuple[float, float]] = None
        self._drag_start_norm: Optional[Tuple[float, float, float, float]] = None
        self._resize_handle: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────

    def set_tool(self, mode: str) -> None:
        self.tool_mode = mode
        self.canvas.config(cursor="crosshair" if mode == "draw" else "arrow")
        if mode == "draw":
            for bv in self._get_current_boxes():
                bv.selected = False
            self._on_selection_changed(None)
            self._on_changed(committed=False)

    def reset_drag_state(self) -> None:
        """Clear in-progress drag/draw state. Called when switching images."""
        self._drag_mode = None
        self._draw_start = None

    def get_selected(self) -> Optional[BoxView]:
        return next((bv for bv in self._get_current_boxes() if bv.selected), None)

    def nudge(self, dx_px: int, dy_px: int) -> None:
        sel = self.get_selected()
        if not sel:
            return
        self._push_undo()
        t = self.transform
        box = sel.box
        box.cx = float(np.clip(
            box.cx + dx_px / t.display_scale / t.current_img_w,
            box.w / 2, 1 - box.w / 2,
        ))
        box.cy = float(np.clip(
            box.cy + dy_px / t.display_scale / t.current_img_h,
            box.h / 2, 1 - box.h / 2,
        ))
        self._on_changed(committed=True)

    def delete_selected(self) -> bool:
        sel = self.get_selected()
        if sel is None:
            return False
        self._push_undo()
        boxes = self._get_current_boxes()
        # In-place mutation so the app's reference stays valid.
        boxes[:] = [bv for bv in boxes if not bv.selected]
        self._on_changed(committed=True)
        return True

    # ── Canvas event handlers ─────────────────────────────────────────────

    def on_click(self, event) -> None:
        cx, cy = float(event.x), float(event.y)

        if self.transform.space_held:
            self.transform.start_pan_drag(cx, cy)
            return

        if self.tool_mode == "draw":
            self._push_undo()
            self._draw_start = (cx, cy)
            self._draw_preview_id = None
            return

        sel = self.get_selected()
        if sel is not None:
            handle = self._hit_test_handles(sel, cx, cy)
            if handle:
                self._push_undo()
                self._drag_mode = "resize"
                self._resize_handle = handle
                self._drag_start_canvas = (cx, cy)
                self._drag_start_norm = (sel.box.cx, sel.box.cy, sel.box.w, sel.box.h)
                return

        boxes = self._get_current_boxes()
        hit: Optional[BoxView] = None
        for bv in reversed(boxes):
            if not self._get_class_enabled(bv.box.class_id):
                continue
            if bv.canvas_x1 <= cx <= bv.canvas_x2 and bv.canvas_y1 <= cy <= bv.canvas_y2:
                hit = bv
                break

        for bv in boxes:
            bv.selected = False

        if hit:
            hit.selected = True
            self._on_selection_changed(hit.box.class_id)
            self._push_undo()
            self._drag_mode = "move"
            self._drag_start_canvas = (cx, cy)
            self._drag_start_norm = (hit.box.cx, hit.box.cy, hit.box.w, hit.box.h)
        else:
            self._on_selection_changed(None)
            self._drag_mode = None

        self._on_changed(committed=False)

    def on_motion(self, event) -> None:
        cx, cy = float(event.x), float(event.y)

        if self.transform.space_held and self.transform.is_pan_dragging:
            if self.transform.update_pan_drag(cx, cy):
                self._on_changed(committed=False)
            return

        if self.tool_mode == "draw" and self._draw_start:
            sx, sy = self._draw_start
            if self._draw_preview_id:
                self.canvas.coords(self._draw_preview_id, sx, sy, cx, cy)
            else:
                self._draw_preview_id = self.canvas.create_rectangle(
                    sx, sy, cx, cy, outline="#ffffff", width=1, dash=(4, 4),
                )
            return

        if self._drag_mode == "move":
            sel = self.get_selected()
            if sel is None or self._drag_start_canvas is None:
                return
            ddx = cx - self._drag_start_canvas[0]
            ddy = cy - self._drag_start_canvas[1]
            t = self.transform
            d_norm_x = ddx / t.display_scale / t.current_img_w
            d_norm_y = ddy / t.display_scale / t.current_img_h
            orig_cx, orig_cy, orig_w, orig_h = self._drag_start_norm
            sel.box.cx = float(np.clip(orig_cx + d_norm_x, orig_w / 2, 1 - orig_w / 2))
            sel.box.cy = float(np.clip(orig_cy + d_norm_y, orig_h / 2, 1 - orig_h / 2))
            self._on_changed(committed=False)
            return

        if self._drag_mode == "resize":
            sel = self.get_selected()
            if sel is None:
                return
            self._apply_resize(sel, cx, cy)
            self._on_changed(committed=False)

    def on_release(self, event) -> None:
        cx, cy = float(event.x), float(event.y)

        if self.tool_mode == "draw" and self._draw_start is not None:
            sx, sy = self._draw_start
            if self._draw_preview_id:
                self.canvas.delete(self._draw_preview_id)
                self._draw_preview_id = None
            self._draw_start = None

            if abs(cx - sx) >= MIN_BOX_DRAW_PX and abs(cy - sy) >= MIN_BOX_DRAW_PX:
                x1c, y1c = min(sx, cx), min(sy, cy)
                x2c, y2c = max(sx, cx), max(sy, cy)
                nx1, ny1 = self.transform.canvas_to_norm_clamped(x1c, y1c)
                nx2, ny2 = self.transform.canvas_to_norm_clamped(x2c, y2c)
                if nx2 > nx1 and ny2 > ny1:
                    new_box = BoundingBox(
                        self._get_draw_class(),
                        (nx1 + nx2) / 2, (ny1 + ny2) / 2,
                        nx2 - nx1, ny2 - ny1,
                        confidence=1.0,
                    )
                    self._get_current_boxes().append(BoxView(box=new_box))
                    self._on_changed(committed=True)
            else:
                self.undo_stack.discard_last(self._get_current_path())
            return

        if self._drag_mode in ("move", "resize"):
            self._on_changed(committed=True)

        self._drag_mode = None
        self._resize_handle = None
        self._drag_start_canvas = None
        self._drag_start_norm = None

        if self.transform.space_held:
            self.transform.end_pan_drag()

    # ── Internals ─────────────────────────────────────────────────────────

    def _push_undo(self) -> None:
        path = self._get_current_path()
        domain = [bv.box for bv in self._get_current_boxes()]
        self.undo_stack.push(path, domain)

    def _hit_test_handles(
        self, bv: BoxView, cx: float, cy: float,
    ) -> Optional[str]:
        half = HANDLE_SIZE // 2
        x1, y1, x2, y2 = bv.canvas_x1, bv.canvas_y1, bv.canvas_x2, bv.canvas_y2
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        positions = {
            "nw": (x1, y1), "n": (mx, y1), "ne": (x2, y1),
            "w":  (x1, my),                "e":  (x2, my),
            "sw": (x1, y2), "s": (mx, y2), "se": (x2, y2),
        }
        for name, (hx, hy) in positions.items():
            if (hx - half) <= cx <= (hx + half) and (hy - half) <= cy <= (hy + half):
                return name
        return None

    def _apply_resize(self, bv: BoxView, cx: float, cy: float) -> None:
        t = self.transform
        orig_cx, orig_cy, orig_w, orig_h = self._drag_start_norm
        img_cx = orig_cx * t.current_img_w
        img_cy = orig_cy * t.current_img_h
        img_w  = orig_w  * t.current_img_w
        img_h  = orig_h  * t.current_img_h
        ccx = img_cx * t.display_scale + t.display_offset_x
        ccy = img_cy * t.display_scale + t.display_offset_y
        hw = img_w * t.display_scale / 2
        hh = img_h * t.display_scale / 2
        ox1, oy1, ox2, oy2 = ccx - hw, ccy - hh, ccx + hw, ccy + hh

        fx1, fy1, fx2, fy2 = ox1, oy1, ox2, oy2
        handle = self._resize_handle
        if "n" in handle: fy1 = cy
        if "s" in handle: fy2 = cy
        if "w" in handle: fx1 = cx
        if "e" in handle: fx2 = cx

        fx1, fx2 = min(fx1, fx2), max(fx1, fx2)
        fy1, fy2 = min(fy1, fy2), max(fy1, fy2)
        if fx2 - fx1 < MIN_BOX_RESIZE_PX: fx2 = fx1 + MIN_BOX_RESIZE_PX
        if fy2 - fy1 < MIN_BOX_RESIZE_PX: fy2 = fy1 + MIN_BOX_RESIZE_PX

        ix1 = max(0.0, min(float(t.current_img_w),
                           (fx1 - t.display_offset_x) / t.display_scale))
        iy1 = max(0.0, min(float(t.current_img_h),
                           (fy1 - t.display_offset_y) / t.display_scale))
        ix2 = max(0.0, min(float(t.current_img_w),
                           (fx2 - t.display_offset_x) / t.display_scale))
        iy2 = max(0.0, min(float(t.current_img_h),
                           (fy2 - t.display_offset_y) / t.display_scale))

        box = bv.box
        box.cx = ((ix1 + ix2) / 2) / t.current_img_w
        box.cy = ((iy1 + iy2) / 2) / t.current_img_h
        box.w  = (ix2 - ix1) / t.current_img_w
        box.h  = (iy2 - iy1) / t.current_img_h
