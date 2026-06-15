import tkinter as tk
from typing import List, Optional

from PIL import Image, ImageTk

from auto_annotator.config import CANVAS_H, CANVAS_W, CLASS_COLORS, HANDLE_SIZE
from auto_annotator.ui.box_view import BoxView
from auto_annotator.ui.canvas_transform import CanvasTransform

HIDDEN = -1
"""Sentinel returned by render() when boxes_hidden is True."""


class CanvasRenderer:
    """Draws the loaded image plus visible box overlays onto a tk.Canvas.

    Holds a reference to the most recent PhotoImage to prevent it from being
    garbage-collected mid-render; otherwise stateless across calls.
    """

    def __init__(self, canvas: tk.Canvas, transform: CanvasTransform):
        self.canvas = canvas
        self.transform = transform
        self._photo_ref: Optional[ImageTk.PhotoImage] = None

    def render(
        self,
        pil_image: Optional[Image.Image],
        box_views: List[BoxView],
        boxes_hidden: bool,
        class_labels: List[str],
        class_enabled: List[bool],
    ) -> int:
        """Render the canvas. Returns the count of visible boxes, or HIDDEN."""
        self.canvas.delete("all")
        if pil_image is None:
            return 0

        cw = self.canvas.winfo_width() or CANVAS_W
        ch = self.canvas.winfo_height() or CANVAS_H
        _, ox, oy, dw, dh = self.transform.fit_image(
            cw, ch, pil_image.width, pil_image.height,
        )
        resized = pil_image.resize((dw, dh), Image.BILINEAR)
        self._photo_ref = ImageTk.PhotoImage(resized)
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._photo_ref)

        if boxes_hidden:
            return HIDDEN

        visible = 0
        for bv in box_views:
            box = bv.box
            if box.class_id >= len(class_enabled):
                continue
            if not class_enabled[box.class_id]:
                continue
            visible += 1
            x1, y1, x2, y2 = self.transform.norm_to_canvas(box.cx, box.cy, box.w, box.h)
            bv.canvas_x1, bv.canvas_y1 = x1, y1
            bv.canvas_x2, bv.canvas_y2 = x2, y2
            color = CLASS_COLORS[box.class_id % len(CLASS_COLORS)]
            lw = 3 if bv.selected else 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=lw)
            label_name = (
                class_labels[box.class_id]
                if box.class_id < len(class_labels)
                else str(box.class_id)
            )
            label_text = f"{label_name} {box.confidence:.2f}"
            lbl_w = len(label_text) * 6 + 6
            lbl_h = 14
            self.canvas.create_rectangle(
                x1, y1 - lbl_h, x1 + lbl_w, y1, fill=color, outline="",
            )
            self.canvas.create_text(
                x1 + 3, y1 - 2, text=label_text, anchor=tk.SW,
                fill="black", font=("TkDefaultFont", 8),
            )
            if bv.selected:
                self._draw_handles(x1, y1, x2, y2)

        return visible

    def _draw_handles(self, x1: float, y1: float, x2: float, y2: float) -> None:
        half = HANDLE_SIZE // 2
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        for hx, hy in [
            (x1, y1), (mx, y1), (x2, y1),
            (x1, my),           (x2, my),
            (x1, y2), (mx, y2), (x2, y2),
        ]:
            self.canvas.create_rectangle(
                hx - half, hy - half, hx + half, hy + half,
                fill="#ffffff", outline="#000000", width=1,
            )
