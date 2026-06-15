# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed
- **Architecture refactor**: `app.py` split from a 1,181-line monolith into a `AnnotatorApp` orchestrator plus seven focused modules under `src/auto_annotator/ui/`:
  - `ui/canvas_transform.py` — zoom, pan, coordinate math
  - `ui/canvas_renderer.py` — image and box overlay drawing
  - `ui/bbox_editor.py` — mouse state machine (draw, select, move, resize, nudge, delete)
  - `ui/class_panel.py` — class filter, per-class confidence, draw-class buttons, selected-box class buttons
  - `ui/file_panel.py` — sidebar file list and navigation
  - `ui/undo.py` — per-image undo stack with depth cap
  - `ui/box_view.py` — UI overlay wrapping a domain `BoundingBox`
- **`BoundingBox` split**: domain class in `models.py` now contains only `class_id`, `cx`, `cy`, `w`, `h`, `confidence`. UI state (`selected`, `canvas_*`) moves to `BoxView` in `ui/box_view.py`.
- Magic-number constants moved from inline literals to `config.py`: `ZOOM_STEP_KEYBOARD`, `ZOOM_STEP_SCROLL`, `MIN_BOX_DRAW_PX`, `MIN_BOX_RESIZE_PX`, `NUDGE_PX`, `CONF_SPINBOX_INCREMENT`, `UNDO_DEPTH_MAX`.

### Added
- 26 new unit tests covering `UndoStack` (10) and `CanvasTransform` (16). Test count: 38 → 64.

## [0.3.0] - 2026-05-13

### Changed
- **Major refactor**: monolithic `annotate_gui.py` split into a Python package under `src/auto_annotator/`. Entry point is now `python -m auto_annotator` (or the `annotate` console script).
- Install and uninstall scripts updated for the new package layout.

### Added
- Modular parser architecture: model output decoders live in `src/auto_annotator/parsers/` behind an `OutputParser` ABC. New architectures plug in by adding a parser and registering it in `PARSER_REGISTRY`.
- Multi-model TensorRT engine support. Engine architecture is auto-detected at load time from output tensor names and shapes.

## [0.2.0] - 2026-04-28

### Added
- RF-DETR model architecture support alongside YOLO11
- Annotation import feature (load existing YOLO `.txt` annotations)
- Hide annotations toggle (Shift key)
- Bulk annotation for all images in one pass
- YOLO11 Medium model engine support

## [0.1.0] - 2026-04-21

### Added
- TensorRT-accelerated YOLO11 inference on Jetson
- YOLO `.txt` and COCO JSON export formats
- Bounding box draw, move, resize, and delete tools
- Undo/redo (20 levels per image)
- Model selection dialog at startup
- Pan/zoom canvas navigation
- Class-based filtering and per-class confidence sliders
- Desktop integration installer for Jetson/JetPack
