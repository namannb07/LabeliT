# Contributing to Auto Annotate

Thanks for your interest. This project targets a narrow niche (Jetson + TensorRT), but most of the codebase is plain Python and you do not need a Jetson to contribute meaningfully.

## What you can work on without a Jetson

Unit-testable layers (no GPU, no TensorRT required):
- `src/auto_annotator/parsers/` — model output decoding (numpy only)
- `src/auto_annotator/store.py` — annotation CRUD, YOLO and COCO import/export
- `src/auto_annotator/utils.py` — label-file parsing
- `src/auto_annotator/config.py` — constants

Layers that need Jetson hardware to test end-to-end:
- `src/auto_annotator/detector.py` — wraps TensorRT/pycuda
- `src/auto_annotator/app.py`, `dialogs.py` — tkinter GUI

If you change the GUI or detector, please flag in your PR that you have or have not tested on hardware.

## Dev setup

You need:
- Python 3.10
- [`uv`](https://github.com/astral-sh/uv) — we use `uv` exclusively, never `pip` directly

```bash
git clone https://github.com/LabeliT/LabeliT.git
cd LabeliT
uv venv --python 3.10
uv pip install -e ".[dev]"
```

On a non-Jetson host, `uv pip install -e ".[dev]"` will try to build `pycuda` and fail without a CUDA toolchain. To work around that for development, install dev tools without resolving runtime deps:

```bash
uv pip install --no-deps -e .
uv pip install pytest pytest-cov ruff numpy opencv-python-headless pillow
```

## Running tests

```bash
uv run pytest
```

Tests marked `@pytest.mark.jetson` are skipped by default. Run them with `uv run pytest -m jetson` on a Jetson device.

## Lint

```bash
uv run ruff check .
uv run ruff format .
```

CI runs `ruff check` on every PR and will fail on lint errors.

## Submitting changes

1. **Open an issue first** for non-trivial changes (new features, refactors, anything > 50 LOC). For typo fixes and small bugs, a direct PR is fine.
2. **Branch off `main`**, push to a fork or a feature branch.
3. **Keep PRs focused** — one logical change per PR.
4. **Add tests** for new logic in `parsers/`, `store.py`, or `utils.py`. If a change is GUI-only or detector-only and not unit-testable, say so in the PR description.
5. **Update `CHANGELOG.md`** under the `[Unreleased]` section if your change is user-visible.
6. **Commit messages**: short imperative subject line. Look at `git log --oneline -20` for the prevailing style.

## Adding a new model architecture

This is the most common contribution shape. Steps:

1. Create `src/auto_annotator/parsers/<name>.py`
2. Subclass `OutputParser` from `parsers/base.py` and implement:
   - `can_parse(output_names, output_shapes)` — class-method probe used by the registry to dispatch
   - `preprocess(img_bgr, net_w, net_h)` — return `(tensor, metadata)`
   - `parse(output_bufs, meta, orig_w, orig_h, conf_thresh)` — return `List[BoundingBox]`
   - `name`, `num_model_classes` properties
3. Add your class to `PARSER_REGISTRY` in `parsers/__init__.py`. Order matters: more-specific `can_parse` checks go first.
4. Add tests under `tests/test_parsers_<name>.py` that feed your `parse()` synthetic buffers and assert decoded boxes.

## Adding a new export format

Add the method to `AnnotationStore` in `store.py`. Do not add export logic to `AnnotatorApp` — the separation is enforced by convention.

## Code of Conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). Be respectful.

## Reporting security issues

See [SECURITY.md](SECURITY.md). Do not file public issues for security problems.
