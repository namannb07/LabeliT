# annotate — Claude project context

TensorRT-accelerated bulk image annotation GUI for Jetson/JetPack Linux. Supports YOLO11 and RF-DETR models. Structured as a Python package under `src/auto_annotator/`.

## Project Layout

```
src/auto_annotator/
    __init__.py       # Package metadata + public API re-exports
    __main__.py       # Entry point: main() function
    config.py         # All constants (paths, thresholds, colors, canvas params)
    models.py         # Domain data classes: ModelType, TRTBuffer, BoundingBox
    utils.py          # load_labels_file() utility
    detector.py       # TRTDetector class — engine/CUDA manager, delegates parsing
    store.py          # AnnotationStore class (CRUD + YOLO/COCO import/export)
    dialogs.py        # ModelSelectionDialog class
    app.py            # AnnotatorApp orchestrator (main GUI window)
    parsers/
        __init__.py   # Parser registry + detect_parser() factory
        base.py       # OutputParser ABC + shared utilities (letterbox, NMS)
        yolo.py       # YoloParser — YOLO v5/v7/v8/v9/v10/v11 family
        detr.py       # DetrParser — RF-DETR (transformer-based)
    ui/
        __init__.py
        box_view.py        # BoxView — UI overlay wrapping a BoundingBox
        canvas_transform.py # CanvasTransform — zoom/pan + coord math
        canvas_renderer.py  # CanvasRenderer — image + box overlay drawing
        bbox_editor.py     # BboxEditor — mouse state machine for draw/move/resize
        class_panel.py     # ClassPanel — class filter / draw buttons / selected-box
        file_panel.py      # FilePanel — sidebar file list + navigation
        undo.py            # UndoStack — per-image undo history
pyproject.toml        # PEP 621 metadata, hatchling build-system
scripts/install.sh    # Desktop integration installer
scripts/uninstall.sh  # Uninstaller
assets/auto_annotate.svg # Application icon
```

## Key Classes

| Class | Module | Role |
|-------|--------|------|
| `TRTDetector` | `detector.py` | Loads a `.engine` file, manages CUDA buffers, runs inference. Delegates output decoding to a matched `OutputParser`. |
| `OutputParser` | `parsers/base.py` | ABC for model-specific output parsers. Subclasses: `YoloParser`, `DetrParser`. |
| `AnnotationStore` | `store.py` | In-memory annotation state; YOLO `.txt` export and COCO JSON export. |
| `BoundingBox` | `models.py` | Pure domain data: `class_id`, `cx`, `cy`, `w`, `h`, `confidence`. Used by parsers, store, exports. |
| `BoxView` | `ui/box_view.py` | UI overlay wrapping a `BoundingBox` with `selected` + cached canvas coords. Never persisted. |
| `CanvasTransform` | `ui/canvas_transform.py` | Owns zoom/pan + coord conversion between normalized, image, and canvas spaces. |
| `CanvasRenderer` | `ui/canvas_renderer.py` | Renders image and box overlays onto a `tk.Canvas`. |
| `BboxEditor` | `ui/bbox_editor.py` | Mouse state machine — click/motion/release dispatch for draw / select / move / resize / hit-test. |
| `ClassPanel` | `ui/class_panel.py` | Class filter checkboxes, per-class confidence spinboxes, draw-class buttons, selected-box class buttons. |
| `FilePanel` | `ui/file_panel.py` | Sidebar file list with annotation-state color coding and arrow-key navigation. |
| `UndoStack` | `ui/undo.py` | Per-image undo history with depth cap (`UNDO_DEPTH_MAX` from config). |
| `ModelSelectionDialog` | `dialogs.py` | Startup `tk.Toplevel`; user picks engine + labels file. |
| `AnnotatorApp` | `app.py` | Root `tk.Tk` orchestrator. Wires the panels and editor together, handles image loading, inference, and export/import dialogs. |

## Runtime Paths (defined in `config.py`)

```
WORK_DIR     = ~/Desktop/auto_annotate
IMAGES_DIR   = ~/Desktop/auto_annotate/input_images
DATASETS_DIR = ~/Desktop/auto_annotate/datasets
```

## Dev Setup

```bash
uv venv --python 3.10
uv pip install -e .
uv run python -m auto_annotator
```

Ensure JetPack TensorRT/CUDA system libs are on `PYTHONPATH` / `LD_LIBRARY_PATH` (JetPack normally sets these).

## Hard Constraints

- **Platform:** JetPack / Jetson Linux only. No cross-platform shims.
- **Python:** 3.10 exactly (JetPack constraint).
- **Package manager:** always `uv` — never `pip` directly.
- **`tensorrt`:** system package from JetPack. Do NOT add it to `pyproject.toml` as a PyPI dep.
- **No async/await:** tkinter's mainloop is incompatible without explicit bridging.

## Important Invariants

- **Undo stack** (`ui/undo.py`): any mutation to `current_boxes` must push a snapshot first via `self.undo_stack.push(path, domain_boxes)` (the editor and app do this automatically before each mutation). Skipping it silently breaks undo. Depth cap is `UNDO_DEPTH_MAX` from `config.py`.
- **`BoundingBox` is domain-only.** UI state (`selected`, canvas coords) lives on `BoxView`. Anything persisted, exported, or returned from parsers/store must be `BoundingBox`. Anything rendered or mouse-tested is `BoxView`.
- **New model architecture:** create a parser in `src/auto_annotator/parsers/`, implement `OutputParser` ABC, and add it to `PARSER_REGISTRY` in `parsers/__init__.py`.
- **New export format:** add to `AnnotationStore` (`store.py`), not `AnnotatorApp`.
- **New constants:** add to `config.py`, not inline.
- **GUI/canvas changes**: drawing goes in `ui/canvas_renderer.py`, mouse logic in `ui/bbox_editor.py`, coord math in `ui/canvas_transform.py`. `app.py` is an orchestrator — it should not gain direct canvas-drawing or mouse-state code.

## What NOT To Do

- `pip install` anything — use `uv`
- Add `tensorrt` to `pyproject.toml`
- Add GUI logic to `AnnotationStore` or export logic to `AnnotatorApp`
