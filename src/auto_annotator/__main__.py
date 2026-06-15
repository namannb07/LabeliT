import sys
import tkinter as tk
import traceback
from tkinter import messagebox

from auto_annotator.app import AnnotatorApp
from auto_annotator.config import (
    COCO_ANNOTATIONS_DIR,
    DATASETS_DIR,
    IMAGES_DIR,
    WORK_DIR,
    YOLO_ANNOTATIONS_DIR,
)
from auto_annotator.detector import TRTDetector
from auto_annotator.dialogs import (
    EmptyStateDialog,
    ExportEngineDialog,
    FrameExtractionDialog,
    LauncherDialog,
    ModelSelectionDialog,
)
from auto_annotator.store import AnnotationStore
from auto_annotator.utils import load_labels_file

_LOG_FILE = WORK_DIR / "crash.log"


def main():
    try:
        _main_impl()
    except Exception:
        msg = traceback.format_exc()
        try:
            _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LOG_FILE.write_text(msg)
        except Exception:
            pass
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Unexpected Error", msg)
            root.destroy()
        except Exception:
            pass
        sys.exit(1)


def _main_impl():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    COCO_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    root.withdraw()

    while True:
        launcher = LauncherDialog(root)

        if launcher.choice is None:
            root.destroy()
            sys.exit(0)

        if launcher.choice == "export":
            ExportEngineDialog(root)
            continue

        if launcher.choice == "extract_frames":
            FrameExtractionDialog(root)
            continue

        break

    def _scan_images():
        return (
            sorted(IMAGES_DIR.glob("*.jpg"))
            + sorted(IMAGES_DIR.glob("*.jpeg"))
            + sorted(IMAGES_DIR.glob("*.png"))
        )

    image_paths = _scan_images()
    while not image_paths:
        empty = EmptyStateDialog(root)
        if empty.outcome == "extract":
            FrameExtractionDialog(root)
            image_paths = _scan_images()
            continue
        if empty.outcome == "retry":
            image_paths = _scan_images()
            continue
        root.destroy()
        sys.exit(0)

    dialog = ModelSelectionDialog(root)
    if not dialog.confirmed:
        root.destroy()
        sys.exit(0)

    try:
        labels = load_labels_file(dialog.labels_path)
    except Exception as e:
        messagebox.showerror("Label Load Error", str(e), parent=None)
        root.destroy()
        sys.exit(1)

    if not labels:
        messagebox.showerror("Label Load Error",
                             f"No labels found in {dialog.labels_path}", parent=None)
        root.destroy()
        sys.exit(1)

    detector = TRTDetector()
    try:
        detector.load(dialog.engine_path, labels)
    except Exception as e:
        messagebox.showerror("Engine Load Error", str(e), parent=None)
        root.destroy()
        sys.exit(1)

    labels = detector.labels

    existing = {l.lower() for l in labels}
    for extra in dialog.extra_labels:
        if extra.lower() not in existing:
            labels.append(extra)
            existing.add(extra.lower())

    print(f"Engine loaded — model: {detector.model_type}, "
          f"input: {detector.net_w}×{detector.net_h}, "
          f"model classes: {detector.num_model_classes}, "
          f"total labels: {len(labels)}")

    store = AnnotationStore()
    for p in image_paths:
        store.load_existing(p)

    root.destroy()
    app = AnnotatorApp(detector, store, image_paths, labels)
    app.mainloop()
    detector.close()


if __name__ == "__main__":
    main()
