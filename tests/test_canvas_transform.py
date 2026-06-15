import pytest

from auto_annotator.ui.canvas_transform import CanvasTransform


def test_init_defaults():
    t = CanvasTransform()
    assert t.zoom == 1.0
    assert t.pan_x == 0.0
    assert t.pan_y == 0.0
    assert not t.is_pan_dragging
    assert not t.space_held


def test_fit_image_square_in_wider_canvas_letterboxes_horizontally():
    t = CanvasTransform()
    scale, ox, oy, dw, dh = t.fit_image(800, 680, 640, 640)
    # min(800/640, 680/640) = 680/640
    expected_scale = 680 / 640
    assert scale == pytest.approx(expected_scale)
    assert dw == 680
    assert dh == 680
    assert ox == pytest.approx((800 - 680) / 2)
    assert oy == pytest.approx(0.0)


def test_fit_image_caches_state_for_coord_conversions():
    t = CanvasTransform()
    t.fit_image(800, 680, 640, 480)
    assert t.current_img_w == 640
    assert t.current_img_h == 480
    assert t.display_scale > 0


def test_norm_to_canvas_to_norm_round_trip_is_identity():
    t = CanvasTransform()
    t.fit_image(800, 680, 640, 480)
    for nx, ny in [(0.0, 0.0), (0.5, 0.5), (0.9, 0.1), (1.0, 1.0)]:
        x1, y1, x2, y2 = t.norm_to_canvas(nx, ny, 0.0, 0.0)
        rx, ry = t.canvas_to_norm_clamped(x1, y1)
        assert rx == pytest.approx(nx, abs=1e-6)
        assert ry == pytest.approx(ny, abs=1e-6)


def test_canvas_to_norm_clamped_clamps_below_zero_to_zero():
    t = CanvasTransform()
    t.fit_image(800, 680, 640, 480)
    x, y = t.canvas_to_norm_clamped(-100, -100)
    assert x == 0.0
    assert y == 0.0


def test_canvas_to_norm_clamped_clamps_above_image_to_one():
    t = CanvasTransform()
    t.fit_image(800, 680, 640, 480)
    x, y = t.canvas_to_norm_clamped(10000, 10000)
    assert x == 1.0
    assert y == 1.0


def test_zoom_at_increases_zoom_factor():
    t = CanvasTransform()
    t.zoom_at(400, 340, 1.5, 800, 680, 640, 480)
    assert t.zoom == pytest.approx(1.5)


def test_zoom_at_clamps_to_max():
    t = CanvasTransform()
    for _ in range(50):
        t.zoom_at(400, 340, 2.0, 800, 680, 640, 480)
    assert t.zoom <= 20.0


def test_zoom_at_clamps_to_min():
    t = CanvasTransform()
    for _ in range(50):
        t.zoom_at(400, 340, 0.5, 800, 680, 640, 480)
    assert t.zoom >= 0.1


def test_zoom_at_anchors_image_point_under_cursor():
    t = CanvasTransform()
    t.fit_image(800, 680, 640, 480)
    # Anchor at canvas point (300, 200)
    cx_anchor, cy_anchor = 300.0, 200.0
    result_before = t.canvas_to_norm_clamped(cx_anchor, cy_anchor)
    t.zoom_at(cx_anchor, cy_anchor, 2.0, 800, 680, 640, 480)
    t.fit_image(800, 680, 640, 480)
    result_after = t.canvas_to_norm_clamped(cx_anchor, cy_anchor)
    assert result_after[0] == pytest.approx(result_before[0], abs=1e-3)
    assert result_after[1] == pytest.approx(result_before[1], abs=1e-3)


def test_reset_view_restores_defaults():
    t = CanvasTransform()
    t.zoom = 2.0
    t.pan_x = 100.0
    t.pan_y = 50.0
    t.reset_view()
    assert t.zoom == 1.0
    assert t.pan_x == 0.0
    assert t.pan_y == 0.0


def test_pan_drag_cycle():
    t = CanvasTransform()
    t.start_pan_drag(100, 100)
    assert t.is_pan_dragging
    changed = t.update_pan_drag(150, 120)
    assert changed
    assert t.pan_x == 50.0
    assert t.pan_y == 20.0
    t.end_pan_drag()
    assert not t.is_pan_dragging


def test_pan_drag_update_without_start_returns_false():
    t = CanvasTransform()
    assert not t.update_pan_drag(100, 100)
    assert t.pan_x == 0.0


def test_space_held_toggle():
    t = CanvasTransform()
    assert not t.space_held
    t.set_space_held(True)
    assert t.space_held
    t.set_space_held(False)
    assert not t.space_held


def test_space_release_ends_active_pan_drag():
    t = CanvasTransform()
    t.start_pan_drag(50, 50)
    t.set_space_held(False)
    assert not t.is_pan_dragging
