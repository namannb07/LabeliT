# LabelIt

[![CI](https://github.com/LabeliT/LabeliT/actions/workflows/ci.yml/badge.svg)](https://github.com/LabeliT/LabeliT/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

TensorRT-accelerated bulk image annotation GUI for Jetson / JetPack Linux.

Manual bounding-box labeling is the slowest part of building an object-detection dataset, and most desktop annotation tools have no inference built in. Auto Annotate uses your locally-built TensorRT engines (YOLO11, RF-DETR) to pre-label images on-device, then puts you in a fast review-and-correct loop. Export to YOLO `.txt` or COCO JSON when you are done. Nothing leaves the machine.

> Built by LabeliT — we build edge-AI systems on Jetson and open-source the tools we use internally.

<!-- Screenshot placeholder. Capture on a Jetson and commit to assets/screenshots/main.png, then uncomment: -->
<!-- ![Auto Annotate](assets/screenshots/main.png) -->

---

## When this is the right tool

Auto Annotate fills a narrow gap. It is for you if:

- You have a Jetson (Nano / Orin / Xavier / AGX) and want to use its spare GPU cycles for labeling.
- You have a TensorRT `.engine` for your domain (or can build one from ONNX with `trtexec`).
- You need data to stay on-device — no cloud uploads, no accounts.
- YOLO `.txt` or COCO JSON is your target export format.

If you do not own a Jetson, you will be better served by [LabelImg](https://github.com/HumanSignal/labelImg), [CVAT](https://github.com/cvat-ai/cvat), or a cloud annotator like Roboflow. We are not trying to compete with those — we just needed something Jetson-native for our own pipelines and figured it might be useful to others in the same spot.

---

## Requirements

**Hardware:** NVIDIA Jetson device with JetPack.

**Software:**
- JetPack (provides TensorRT, CUDA, cuDNN system libraries)
- Python 3.10
- [`uv`](https://github.com/astral-sh/uv) package manager

**Python dependencies** (installed automatically):
- `pycuda`, `opencv-python-headless`, `pillow`, `numpy`
- `tensorrt` — comes from the JetPack system install, **not** PyPI

---

## Installation

```bash
git clone https://github.com/LabeliT/LabeliT.git
cd LabeliT
bash scripts/install.sh
```

This copies the app to `~/.local/share/auto_annotate`, creates a `uv`-managed virtualenv, installs Python dependencies, registers a `.desktop` entry in the system app menu, and creates the runtime work directories.

---

## Model Setup

TensorRT `.engine` files are device-specific — they must be built on the target Jetson. Engine files are not part of this repo.

1. Start with an ONNX model.
2. Convert to a `.engine` using `trtexec`:
   ```bash
   trtexec --onnx=yolo11_ppe.onnx --saveEngine=yolo11_ppe.engine --fp16
   ```
3. Place the `.engine` and a `labels.txt` (one class name per line) into `~/Desktop/auto_annotate/model/`.
4. At launch, a dialog prompts for the engine file path and labels file path.

**Supported architectures:** YOLO11 (single-output engines, also compatible with YOLOv5/v7/v8/v9/v10 layouts), RF-DETR (two-output transformer engines with `dets` and `labels` tensors).

---

## Usage

1. Drop images into `~/Desktop/auto_annotate/input_images/`.
2. Launch from the app menu or:
   ```bash
   ~/.local/share/auto_annotate/launch.sh
   ```
3. Select the `.engine` file and `labels.txt` in the startup dialog.
4. Click an image in the sidebar to open it.
5. Use **Auto-Annotate** to run inference on the current image, or **Bulk Annotate** to run it across every image.
6. Review predictions — draw, move, resize, or delete bounding boxes manually.
7. Export via the **Export** button (YOLO or COCO).

---

## Project Structure

```
src/auto_annotator/
    __init__.py         # Package metadata + public API re-exports
    __main__.py         # Entry point: main()
    config.py           # Constants (paths, thresholds, colors, canvas params)
    models.py           # Domain data: ModelType, TRTBuffer, BoundingBox
    utils.py            # load_labels_file()
    detector.py         # TRTDetector — engine/CUDA manager, delegates parsing
    store.py            # AnnotationStore — CRUD + YOLO/COCO import/export
    dialogs.py          # ModelSelectionDialog
    app.py              # AnnotatorApp — orchestrator (image loading, inference, export)
    parsers/
        __init__.py     # Parser registry + detect_parser() factory
        base.py         # OutputParser ABC + shared utilities (letterbox, NMS)
        yolo.py         # YoloParser — YOLO v5/v7/v8/v9/v10/v11 family
        detr.py         # DetrParser — RF-DETR
    ui/
        box_view.py        # BoxView — UI overlay wrapping a BoundingBox
        canvas_transform.py # zoom / pan / coord math
        canvas_renderer.py  # image + box overlay drawing
        bbox_editor.py     # mouse state machine (draw / select / move / resize)
        class_panel.py     # class filter, draw buttons, selected-box buttons
        file_panel.py      # sidebar file list + navigation
        undo.py            # per-image undo history
tests/                  # Unit tests for parsers, store, utils, undo, transform
scripts/install.sh      # Desktop integration installer
scripts/uninstall.sh    # Uninstaller
assets/                 # Application icon and screenshots
```

**Runtime directories** (created by the installer):

```
~/.local/share/auto_annotate/     # install destination
~/Desktop/auto_annotate/
  input_images/                   # source images go here
  datasets/                       # exported datasets land here
  model/                          # place .engine and labels.txt here
```

---

## Development

```bash
uv venv --python 3.10
uv pip install -e ".[dev]"
uv run python -m auto_annotator
```

Run the test suite (Jetson-required tests are skipped automatically on non-Jetson hosts):

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for a full dev-setup walkthrough, including how to develop on a non-Jetson host.

---

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Select tool | `S` |
| Draw tool | `D` |
| Delete selected box | `Delete` / `Backspace` |
| Undo | `Ctrl+Z` |
| Nudge selected box | Arrow keys |
| Zoom in | `+` / `=` / scroll up |
| Zoom out | `-` / scroll down |
| Reset zoom | `0` |
| Pan (while held) | `Space` + drag |
| Pan | Middle-mouse drag |
| Hide annotations | `Shift` (hold) |

---

## Configuration

| Setting | Value |
|---------|-------|
| Confidence threshold | Slider in GUI, range 0.05–0.95, default **0.25** |
| NMS IoU threshold | Hardcoded at **0.45** |
| Undo history depth | **20** levels per image |

---

## Export Formats

**YOLO** — one `.txt` per image with normalized `class cx cy w h` per line, plus `classes.txt` and `data.yaml`. Written to:

```
~/Desktop/auto_annotate/datasets/<name>_yolo/
```

**COCO JSON** — single `annotations.json` in COCO format. Written to:

```
~/Desktop/auto_annotate/datasets/<name>_coco/
```

Importing existing YOLO or COCO annotations is supported via the **Import** button — class IDs are remapped against the currently-loaded labels file; unmatched classes are reported and skipped.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorrt'`** — JetPack's TensorRT lives outside the venv `site-packages`. The installer normally bridges this with a `.pth` file. If it did not, confirm `/usr/lib/python3.10/dist-packages/tensorrt` exists and re-run `bash scripts/install.sh`.

**Inference returns no boxes, or boxes in the wrong place** — almost always one of:
- The `.engine` was built on a different Jetson model. Engines are not portable across e.g. Orin Nano ↔ AGX Orin. Rebuild with `trtexec` on the target device.
- `labels.txt` line count does not match the model's class count. Run `wc -l labels.txt` and compare to your training class count.

**`pycuda` build fails during `uv sync`** — ensure `CUDA_ROOT=/usr/local/cuda` is set (the installer does this for you).

**GUI does not appear over SSH** — Auto Annotate needs an X11 session. Run it from the device directly, or use `ssh -X` with an X server on the client.

---

## Roadmap

- Polygon and segmentation-mask annotation
- Active-learning loop: sort images by lowest-confidence predictions
- Pre-built engines for common architectures (YOLOv8/v9/v10, YOLOX)
- Headless / batch-only mode for CI dataset regeneration

Open an issue if you have a specific request.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The parser, store, and utils layers are unit-testable on any machine — those are the easiest places to start without a Jetson.

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability-disclosure policy.

## Uninstall

```bash
bash scripts/uninstall.sh
```

Removes `~/.local/share/auto_annotate` and the `.desktop` entry. User data in `~/Desktop/auto_annotate/` is preserved.

## License

Apache-2.0. See [LICENSE](LICENSE).
