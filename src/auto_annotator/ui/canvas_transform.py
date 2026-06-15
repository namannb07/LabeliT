from typing import Optional, Tuple

from auto_annotator.config import MAX_ZOOM, MIN_ZOOM


class CanvasTransform:
    """Owns zoom/pan state and the coordinate math between image, normalized,
    and canvas-pixel spaces.

    Knows nothing about tkinter. Callers pass canvas dimensions and image
    dimensions; the transform returns offsets/scales for them to render with.
    """

    def __init__(self):
        self.zoom: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

        # Cached from the most recent fit_image() — used by coord conversions
        self.display_scale: float = 1.0
        self.display_offset_x: float = 0.0
        self.display_offset_y: float = 0.0
        self.current_img_w: int = 1
        self.current_img_h: int = 1

        self._pan_drag_start: Optional[Tuple[float, float]] = None
        self._pan_drag_origin: Optional[Tuple[float, float]] = None
        self._space_held: bool = False

    # ── View state ────────────────────────────────────────────────────────

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    # ── Fit + cached display transform ────────────────────────────────────

    def fit_image(
        self, canvas_w: float, canvas_h: float, img_w: int, img_h: int,
    ) -> Tuple[float, float, float, int, int]:
        """Compute the display transform for an image at current zoom/pan.

        Updates cached display_scale / display_offset_x / display_offset_y /
        current_img_w / current_img_h so that subsequent coord conversions
        can use the same transform without re-passing the inputs.

        Returns (scale, offset_x, offset_y, draw_w, draw_h).
        """
        fit_scale = min(canvas_w / img_w, canvas_h / img_h)
        effective_scale = fit_scale * self.zoom
        ox = (canvas_w - img_w * effective_scale) / 2 + self.pan_x
        oy = (canvas_h - img_h * effective_scale) / 2 + self.pan_y
        dw = max(1, int(img_w * effective_scale))
        dh = max(1, int(img_h * effective_scale))
        self.display_scale = effective_scale
        self.display_offset_x = ox
        self.display_offset_y = oy
        self.current_img_w = img_w
        self.current_img_h = img_h
        return effective_scale, ox, oy, dw, dh

    # ── Coord conversion (uses cached display transform) ──────────────────

    def norm_to_canvas(
        self, cx: float, cy: float, w: float, h: float,
    ) -> Tuple[float, float, float, float]:
        img_cx = cx * self.current_img_w
        img_cy = cy * self.current_img_h
        img_w = w * self.current_img_w
        img_h = h * self.current_img_h
        ccx = img_cx * self.display_scale + self.display_offset_x
        ccy = img_cy * self.display_scale + self.display_offset_y
        hw = img_w * self.display_scale / 2
        hh = img_h * self.display_scale / 2
        return ccx - hw, ccy - hh, ccx + hw, ccy + hh

    def canvas_to_norm_clamped(self, cx: float, cy: float) -> Tuple[float, float]:
        img_x = (cx - self.display_offset_x) / self.display_scale
        img_y = (cy - self.display_offset_y) / self.display_scale
        img_x = max(0.0, min(float(self.current_img_w), img_x))
        img_y = max(0.0, min(float(self.current_img_h), img_y))
        return img_x / self.current_img_w, img_y / self.current_img_h

    # ── Zoom ──────────────────────────────────────────────────────────────

    def zoom_at(
        self, canvas_x: float, canvas_y: float, factor: float,
        canvas_w: float, canvas_h: float, img_w: int, img_h: int,
    ) -> None:
        """Zoom by `factor` around the cursor at (canvas_x, canvas_y).

        Adjusts pan so the image pixel under the cursor stays under the cursor.
        Clamped to [MIN_ZOOM, MAX_ZOOM] from config.
        """
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        fit = min(canvas_w / img_w, canvas_h / img_h)
        old_eff = fit * self.zoom
        img_x = (canvas_x - (canvas_w - img_w * old_eff) / 2 - self.pan_x) / old_eff
        img_y = (canvas_y - (canvas_h - img_h * old_eff) / 2 - self.pan_y) / old_eff
        new_eff = fit * new_zoom
        self.pan_x = canvas_x - img_x * new_eff - (canvas_w - img_w * new_eff) / 2
        self.pan_y = canvas_y - img_y * new_eff - (canvas_h - img_h * new_eff) / 2
        self.zoom = new_zoom

    # ── Pan (mouse-drag) ──────────────────────────────────────────────────

    def start_pan_drag(self, canvas_x: float, canvas_y: float) -> None:
        self._pan_drag_start = (canvas_x, canvas_y)
        self._pan_drag_origin = (self.pan_x, self.pan_y)

    def update_pan_drag(self, canvas_x: float, canvas_y: float) -> bool:
        """Apply drag delta. Returns True if state changed (caller should redraw)."""
        if self._pan_drag_start is None or self._pan_drag_origin is None:
            return False
        dx = canvas_x - self._pan_drag_start[0]
        dy = canvas_y - self._pan_drag_start[1]
        self.pan_x = self._pan_drag_origin[0] + dx
        self.pan_y = self._pan_drag_origin[1] + dy
        return True

    def end_pan_drag(self) -> None:
        self._pan_drag_start = None
        self._pan_drag_origin = None

    @property
    def is_pan_dragging(self) -> bool:
        return self._pan_drag_start is not None

    # ── Space-bar pan modifier ────────────────────────────────────────────

    def set_space_held(self, held: bool) -> None:
        self._space_held = held
        if not held:
            self.end_pan_drag()

    @property
    def space_held(self) -> bool:
        return self._space_held
