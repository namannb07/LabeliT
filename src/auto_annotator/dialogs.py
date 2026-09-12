import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

from auto_annotator.config import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_SPLIT_TEST,
    DEFAULT_SPLIT_TRAIN,
    DEFAULT_SPLIT_VAL,
    FRAME_EXTRACT_POLL_MS,
    FRAME_EXTRACT_VIDEO_EXTS,
    IMAGES_DIR,
    MIN_IMAGES_FOR_SPLIT,
    MODELS_DIR,
)
from auto_annotator.ui.tooltip import Tooltip


class ModelSelectionDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.title("Select ONNX Model")
        self.resizable(False, False)
        self.configure(bg="#1e1e1e")
        self.grab_set()

        self.model_path: Optional[Path] = None
        self.labels_path: Optional[Path] = None
        self.extra_labels: List[str] = []
        self.confirmed: bool = False

        self._model_var = tk.StringVar()
        self._labels_var = tk.StringVar()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.update_idletasks()
        pw = parent.winfo_screenwidth()
        ph = parent.winfo_screenheight()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry(f"+{(pw - w) // 2}+{(ph - h) // 2}")

        self.wait_window(self)

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        tk.Label(self, text="LabeliT", bg="#007acc", fg="white",
                 font=("TkDefaultFont", 12, "bold"), pady=10).pack(fill=tk.X)
        tk.Label(self, text="Select an ONNX model file and a labels file to continue.",
                 bg="#1e1e1e", fg="#aaaaaa", font=("TkDefaultFont", 9),
                 pady=6).pack(fill=tk.X, padx=12)

        model_frame = tk.LabelFrame(self, text=" ONNX Model (.onnx) ",
                                    bg="#1e1e1e", fg="#aaaaaa",
                                    font=("TkDefaultFont", 8))
        model_frame.pack(fill=tk.X, **pad)
        model_row = tk.Frame(model_frame, bg="#1e1e1e")
        model_row.pack(fill=tk.X, padx=6, pady=6)
        tk.Entry(model_row, textvariable=self._model_var, state="readonly",
                 bg="#2d2d2d", fg="#d4d4d4", readonlybackground="#2d2d2d",
                 width=48, relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(model_row, text="Browse…", command=self._browse_model,
                  bg="#3c3c3c", fg="#d4d4d4", relief=tk.FLAT,
                  activebackground="#505050", cursor="hand2").pack(side=tk.LEFT)

        lbl_frame = tk.LabelFrame(self, text=" Labels File (.txt) (Optional) ",
                                  bg="#1e1e1e", fg="#aaaaaa",
                                  font=("TkDefaultFont", 8))
        lbl_frame.pack(fill=tk.X, **pad)
        lbl_row = tk.Frame(lbl_frame, bg="#1e1e1e")
        lbl_row.pack(fill=tk.X, padx=6, pady=6)
        tk.Entry(lbl_row, textvariable=self._labels_var, state="readonly",
                 bg="#2d2d2d", fg="#d4d4d4", readonlybackground="#2d2d2d",
                 width=48, relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(lbl_row, text="Browse…", command=self._browse_labels,
                  bg="#3c3c3c", fg="#d4d4d4", relief=tk.FLAT,
                  activebackground="#505050", cursor="hand2").pack(side=tk.LEFT)

        extra_frame = tk.LabelFrame(self, text=" Extra Classes (manual-only) ",
                                    bg="#1e1e1e", fg="#aaaaaa",
                                    font=("TkDefaultFont", 8))
        extra_frame.pack(fill=tk.X, **pad)
        tk.Label(extra_frame, text="Comma-separated names (e.g. gloves, boots):",
                 bg="#1e1e1e", fg="#888888",
                 font=("TkDefaultFont", 8)).pack(anchor=tk.W, padx=6, pady=(4, 0))
        self._extra_entry = tk.Entry(extra_frame, bg="#2d2d2d", fg="#d4d4d4",
                                     insertbackground="#d4d4d4", width=48,
                                     relief=tk.FLAT)
        self._extra_entry.pack(fill=tk.X, padx=6, pady=6)

        btn_row = tk.Frame(self, bg="#1e1e1e")
        btn_row.pack(fill=tk.X, padx=12, pady=(4, 12))
        self._ok_btn = tk.Button(btn_row, text="OK", command=self._on_ok,
                                 bg="#0e639c", fg="white", relief=tk.FLAT,
                                 activebackground="#1177bb", width=10,
                                 cursor="hand2", state=tk.DISABLED)
        self._ok_btn.pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(btn_row, text="Cancel", command=self._on_cancel,
                  bg="#3c3c3c", fg="#d4d4d4", relief=tk.FLAT,
                  activebackground="#505050", width=10,
                  cursor="hand2").pack(side=tk.RIGHT)

        self._model_var.trace_add("write", self._update_ok_btn)
        self._labels_var.trace_add("write", self._update_ok_btn)

    def _update_ok_btn(self, *_):
        if self._model_var.get():
            self._ok_btn.config(state=tk.NORMAL)
        else:
            self._ok_btn.config(state=tk.DISABLED)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select ONNX Model",
            initialdir=MODELS_DIR if MODELS_DIR.exists() else Path.home(),
            filetypes=[("ONNX Model", "*.onnx"), ("All files", "*.*")],
        )
        if path:
            self._model_var.set(path)

    def _browse_labels(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select Labels File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self._labels_var.set(path)

    def _on_ok(self):
        mp = Path(self._model_var.get())
        if not mp.exists():
            messagebox.showerror("Error", f"Model file not found:\n{mp}", parent=self)
            return

        lp_str = self._labels_var.get().strip()
        if lp_str:
            lp = Path(lp_str)
            if not lp.exists():
                messagebox.showerror("Error", f"Labels file not found:\n{lp}", parent=self)
                return
            self.labels_path = lp
        else:
            self.labels_path = None

        self.model_path = mp
        raw = self._extra_entry.get().strip()
        if raw:
            self.extra_labels = [c.strip() for c in raw.split(",") if c.strip()]
        self.confirmed = True
        self.destroy()

    def _on_cancel(self):
        self.confirmed = False
        self.destroy()


class LauncherDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.title("LabeliT")
        self.resizable(False, False)
        self.configure(bg="#1e1e1e")
        self.grab_set()

        self.choice: Optional[str] = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.update_idletasks()
        pw = parent.winfo_screenwidth()
        ph = parent.winfo_screenheight()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry(f"+{(pw - w) // 2}+{(ph - h) // 2}")

        self.wait_window(self)

    def _build_ui(self):
        tk.Label(
            self, text="LabeliT", bg="#007acc", fg="white",
            font=("TkDefaultFont", 12, "bold"), pady=10,
        ).pack(fill=tk.X)
        tk.Label(
            self, text="What would you like to do?",
            bg="#1e1e1e", fg="#d4d4d4", font=("TkDefaultFont", 10, "bold"),
            pady=4,
        ).pack(fill=tk.X, padx=12, pady=(8, 0))
        tk.Label(
            self,
            text="You can switch between these tools later from the Tools menu.",
            bg="#1e1e1e", fg="#888888", font=("TkDefaultFont", 8),
        ).pack(fill=tk.X, padx=12, pady=(0, 6))

        extract_btn = tk.Button(
            self, text="① Frame Extraction\nTurn a video into still images you can annotate",
            command=self._on_extract_frames,
            bg="#2d2d2d", fg="#ffffff", activebackground="#3c3c3c",
            activeforeground="#ffffff", relief=tk.FLAT, cursor="hand2",
            anchor="w", justify="left", padx=14, pady=10,
            font=("TkDefaultFont", 10),
        )
        extract_btn.pack(fill=tk.X, padx=12, pady=(6, 0))
        Tooltip(
            extract_btn,
            "Use this if you have a video (.mp4/.mkv/.webm) and need to create individual images to annotate.",
        )

        annotate_btn = tk.Button(
            self, text="② Start Annotation\nAuto-detect and label images using an ONNX model",
            command=self._on_annotate,
            bg="#2d2d2d", fg="#ffffff", activebackground="#3c3c3c",
            activeforeground="#ffffff", relief=tk.FLAT, cursor="hand2",
            anchor="w", justify="left", padx=14, pady=10,
            font=("TkDefaultFont", 10),
        )
        annotate_btn.pack(fill=tk.X, padx=12, pady=(6, 0))
        Tooltip(
            annotate_btn,
            "Open the main annotator. You'll be asked to pick your .onnx model file and labels.txt, then your images load automatically.",
        )

        btn_row = tk.Frame(self, bg="#1e1e1e")
        btn_row.pack(fill=tk.X, padx=12, pady=(4, 12))
        tk.Button(
            btn_row, text="Cancel", command=self._on_cancel,
            bg="#3c3c3c", fg="#d4d4d4", relief=tk.FLAT,
            activebackground="#505050", width=10, cursor="hand2",
        ).pack(side=tk.RIGHT)

    def _on_annotate(self):
        self.choice = "annotate"
        self.destroy()

    def _on_extract_frames(self):
        self.choice = "extract_frames"
        self.destroy()

    def _on_cancel(self):
        self.choice = None
        self.destroy()



class SplitConfigDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, n_annotated: int):
        super().__init__(parent)
        self.title("Dataset Split Configuration")
        self.resizable(False, False)
        self.configure(bg="#1e1e1e")
        self.grab_set()

        self._n_annotated = n_annotated
        self.confirmed: bool = False
        self.splits_enabled: bool = True
        self.train_pct: int = DEFAULT_SPLIT_TRAIN
        self.val_pct: int = DEFAULT_SPLIT_VAL
        self.test_pct: int = DEFAULT_SPLIT_TEST
        self.seed: int = DEFAULT_SPLIT_SEED

        self._split_var = tk.BooleanVar(value=True)
        self._train_var = tk.StringVar(value=str(DEFAULT_SPLIT_TRAIN))
        self._val_var = tk.StringVar(value=str(DEFAULT_SPLIT_VAL))
        self._test_var = tk.StringVar(value=str(DEFAULT_SPLIT_TEST))
        self._seed_var = tk.StringVar(value=str(DEFAULT_SPLIT_SEED))

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._train_var.trace_add("write", self._on_ratio_change)
        self._val_var.trace_add("write", self._on_ratio_change)
        self._test_var.trace_add("write", self._on_ratio_change)
        self._on_ratio_change()

        self.update_idletasks()
        pw = parent.winfo_screenwidth()
        ph = parent.winfo_screenheight()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry(f"+{(pw - w) // 2}+{(ph - h) // 2}")

        self.wait_window(self)

    def _build_ui(self):
        tk.Label(
            self, text="Dataset Split", bg="#007acc", fg="white",
            font=("TkDefaultFont", 12, "bold"), pady=10,
        ).pack(fill=tk.X)

        self._split_chk = tk.Checkbutton(
            self, text="Enable train / val / test split",
            variable=self._split_var, command=self._on_split_toggle,
            bg="#1e1e1e", fg="#d4d4d4", selectcolor="#2d2d2d",
            activebackground="#1e1e1e", activeforeground="#d4d4d4",
            font=("TkDefaultFont", 10),
        )
        self._split_chk.pack(anchor=tk.W, padx=14, pady=(10, 4))

        self._ratio_frame = tk.LabelFrame(
            self, text=" Split Ratios (%) ",
            bg="#1e1e1e", fg="#aaaaaa", font=("TkDefaultFont", 8),
        )
        self._ratio_frame.pack(fill=tk.X, padx=12, pady=6)

        vcmd = (self.register(self._validate_pct), "%P")

        grid = tk.Frame(self._ratio_frame, bg="#1e1e1e")
        grid.pack(padx=10, pady=8)

        self._spinboxes = []
        for row, (label, var) in enumerate([
            ("Train:", self._train_var),
            ("Val:", self._val_var),
            ("Test:", self._test_var),
        ]):
            tk.Label(
                grid, text=label, bg="#1e1e1e", fg="#d4d4d4",
                font=("TkDefaultFont", 10), width=6, anchor=tk.E,
            ).grid(row=row, column=0, padx=(0, 6), pady=3)
            sb = tk.Spinbox(
                grid, from_=0, to=100, increment=5, width=6,
                textvariable=var, bg="#2d2d2d", fg="#d4d4d4",
                insertbackground="#d4d4d4", buttonbackground="#3c3c3c",
                relief=tk.FLAT, font=("TkDefaultFont", 10),
                validate="key", validatecommand=vcmd,
            )
            sb.grid(row=row, column=1, padx=(0, 4), pady=3)
            self._spinboxes.append(sb)
            tk.Label(
                grid, text="%", bg="#1e1e1e", fg="#888888",
                font=("TkDefaultFont", 9),
            ).grid(row=row, column=2, pady=3)

        self._total_lbl = tk.Label(
            self._ratio_frame, text="Total: 100%",
            bg="#1e1e1e", fg="#44dd44", font=("TkDefaultFont", 10, "bold"),
        )
        self._total_lbl.pack(pady=(0, 8))

        seed_frame = tk.Frame(self, bg="#1e1e1e")
        seed_frame.pack(fill=tk.X, padx=14, pady=(2, 4))
        tk.Label(
            seed_frame, text="Seed:", bg="#1e1e1e", fg="#d4d4d4",
            font=("TkDefaultFont", 9),
        ).pack(side=tk.LEFT)
        self._seed_entry = tk.Entry(
            seed_frame, textvariable=self._seed_var, width=8,
            bg="#2d2d2d", fg="#d4d4d4", insertbackground="#d4d4d4",
            relief=tk.FLAT, font=("TkDefaultFont", 9),
        )
        self._seed_entry.pack(side=tk.LEFT, padx=(6, 4))
        tk.Label(
            seed_frame, text="(for reproducibility)", bg="#1e1e1e", fg="#888888",
            font=("TkDefaultFont", 8),
        ).pack(side=tk.LEFT)

        self._preview_lbl = tk.Label(
            self, text="", bg="#1e1e1e", fg="#d4d4d4",
            font=("TkDefaultFont", 9), justify=tk.LEFT,
        )
        self._preview_lbl.pack(padx=14, pady=(6, 2), anchor=tk.W)

        self._warning_lbl = tk.Label(
            self, text="", bg="#1e1e1e", fg="#ffaa00",
            font=("TkDefaultFont", 8), justify=tk.LEFT,
        )
        self._warning_lbl.pack(padx=14, pady=(0, 4), anchor=tk.W)

        btn_row = tk.Frame(self, bg="#1e1e1e")
        btn_row.pack(fill=tk.X, padx=12, pady=(4, 12))
        self._ok_btn = tk.Button(
            btn_row, text="Export", command=self._on_ok,
            bg="#0e639c", fg="white", relief=tk.FLAT,
            activebackground="#1177bb", width=10, cursor="hand2",
        )
        self._ok_btn.pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(
            btn_row, text="Cancel", command=self._on_cancel,
            bg="#3c3c3c", fg="#d4d4d4", relief=tk.FLAT,
            activebackground="#505050", width=10, cursor="hand2",
        ).pack(side=tk.RIGHT)

    @staticmethod
    def _validate_pct(value: str) -> bool:
        if value == "":
            return True
        try:
            v = int(value)
            return 0 <= v <= 100
        except ValueError:
            return False

    def _on_split_toggle(self):
        enabled = self._split_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        for sb in self._spinboxes:
            sb.config(state=state)
        self._seed_entry.config(state=state)
        self._on_ratio_change()

    def _on_ratio_change(self, *_args):
        enabled = self._split_var.get()
        if not enabled:
            self._total_lbl.config(text="Split disabled", fg="#888888")
            self._preview_lbl.config(text=f"{self._n_annotated} annotated images (no split)")
            self._warning_lbl.config(text="")
            self._ok_btn.config(state=tk.NORMAL)
            return

        train = self._safe_int(self._train_var)
        val = self._safe_int(self._val_var)
        test = self._safe_int(self._test_var)
        total = train + val + test

        if total == 100:
            self._total_lbl.config(text=f"Total: {total}%", fg="#44dd44")
            self._ok_btn.config(state=tk.NORMAL)
        else:
            self._total_lbl.config(text=f"Total: {total}% (must be 100%)", fg="#ff4444")
            self._ok_btn.config(state=tk.DISABLED)

        n = self._n_annotated
        n_test = round(n * test / 100) if test > 0 else 0
        n_val = round(n * val / 100) if val > 0 else 0
        n_train = n - n_val - n_test

        parts = []
        if train > 0:
            parts.append(f"Train: {n_train}")
        if val > 0:
            parts.append(f"Val: {n_val}")
        if test > 0:
            parts.append(f"Test: {n_test}")
        self._preview_lbl.config(
            text=f"{n} annotated images  →  " + "  |  ".join(parts) if parts else ""
        )

        warnings = []
        if n < MIN_IMAGES_FOR_SPLIT:
            warnings.append(
                f"Only {n} image{'s' if n != 1 else ''} — some splits will be empty."
            )
        else:
            empty = []
            if train > 0 and n_train == 0:
                empty.append("train")
            if val > 0 and n_val == 0:
                empty.append("val")
            if test > 0 and n_test == 0:
                empty.append("test")
            if empty:
                warnings.append(
                    f"Not enough images: {', '.join(empty)} "
                    f"split{'s' if len(empty) > 1 else ''} will be empty."
                )
        self._warning_lbl.config(text="\n".join(warnings))

    @staticmethod
    def _safe_int(var: tk.StringVar) -> int:
        try:
            return int(var.get())
        except ValueError:
            return 0

    def _on_cancel(self):
        self.destroy()

    def _on_ok(self):
        self.splits_enabled = self._split_var.get()
        if self.splits_enabled:
            self.train_pct = self._safe_int(self._train_var)
            self.val_pct = self._safe_int(self._val_var)
            self.test_pct = self._safe_int(self._test_var)
            try:
                self.seed = int(self._seed_var.get())
            except ValueError:
                messagebox.showerror(
                    "Invalid Seed", "Seed must be an integer.", parent=self,
                )
                return
        self.confirmed = True
        self.destroy()


class FrameExtractionDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.title("Frame Extraction")
        self.resizable(False, False)
        self.configure(bg="#1e1e1e")
        self.grab_set()

        self._video_path: Optional[Path] = None
        self._video_info = None
        self._indices: List[int] = []
        self._worker: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._status_queue: queue.Queue = queue.Queue()
        self._poll_id: Optional[str] = None

        self._video_var = tk.StringVar()
        self._skip_var = tk.StringVar(value="0")
        self._cap_enabled_var = tk.BooleanVar(value=False)
        self._cap_var = tk.StringVar(value="")
        self._output_var = tk.StringVar(value=str(IMAGES_DIR))

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._skip_var.trace_add("write", self._on_inputs_changed)
        self._cap_var.trace_add("write", self._on_inputs_changed)

        self.update_idletasks()
        pw = parent.winfo_screenwidth()
        ph = parent.winfo_screenheight()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry(f"+{(pw - w) // 2}+{(ph - h) // 2}")

        self.wait_window(self)

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        tk.Label(
            self, text="Frame Extraction", bg="#007acc", fg="white",
            font=("TkDefaultFont", 12, "bold"), pady=10,
        ).pack(fill=tk.X)
        tk.Label(
            self, text="Extract frames from a video into an images folder.",
            bg="#1e1e1e", fg="#aaaaaa", font=("TkDefaultFont", 9), pady=6,
        ).pack(fill=tk.X, padx=12)

        vid_frame = tk.LabelFrame(
            self, text=" Video File (.mp4 / .mkv / .webm) ",
            bg="#1e1e1e", fg="#aaaaaa", font=("TkDefaultFont", 8),
        )
        vid_frame.pack(fill=tk.X, **pad)
        vid_row = tk.Frame(vid_frame, bg="#1e1e1e")
        vid_row.pack(fill=tk.X, padx=6, pady=6)
        tk.Entry(
            vid_row, textvariable=self._video_var, state="readonly",
            bg="#2d2d2d", fg="#d4d4d4", readonlybackground="#2d2d2d",
            width=48, relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self._browse_vid_btn = tk.Button(
            vid_row, text="Browse…", command=self._browse_video,
            bg="#3c3c3c", fg="#d4d4d4", relief=tk.FLAT,
            activebackground="#505050", cursor="hand2",
        )
        self._browse_vid_btn.pack(side=tk.LEFT)

        info_frame = tk.LabelFrame(
            self, text=" Video Info ",
            bg="#1e1e1e", fg="#aaaaaa", font=("TkDefaultFont", 8),
        )
        info_frame.pack(fill=tk.X, **pad)
        self._resolution_lbl = tk.Label(
            info_frame, text="Resolution: —",
            bg="#1e1e1e", fg="#d4d4d4", font=("TkDefaultFont", 9), anchor=tk.W,
        )
        self._resolution_lbl.pack(fill=tk.X, padx=6, pady=(4, 0))
        self._total_lbl = tk.Label(
            info_frame, text="Total frames: —",
            bg="#1e1e1e", fg="#d4d4d4", font=("TkDefaultFont", 9), anchor=tk.W,
        )
        self._total_lbl.pack(fill=tk.X, padx=6)
        self._fps_lbl = tk.Label(
            info_frame, text="FPS: —    Duration: —",
            bg="#1e1e1e", fg="#d4d4d4", font=("TkDefaultFont", 9), anchor=tk.W,
        )
        self._fps_lbl.pack(fill=tk.X, padx=6)
        self._status_lbl = tk.Label(
            info_frame, text="Select a video to begin.",
            bg="#1e1e1e", fg="#888888", font=("TkDefaultFont", 9), anchor=tk.W,
        )
        self._status_lbl.pack(fill=tk.X, padx=6, pady=(0, 6))

        skip_frame = tk.LabelFrame(
            self, text=" Skip Frames ",
            bg="#1e1e1e", fg="#aaaaaa", font=("TkDefaultFont", 8),
        )
        skip_frame.pack(fill=tk.X, **pad)
        skip_row = tk.Frame(skip_frame, bg="#1e1e1e")
        skip_row.pack(fill=tk.X, padx=6, pady=6)
        tk.Label(
            skip_row, text="Skip between captures:",
            bg="#1e1e1e", fg="#d4d4d4", font=("TkDefaultFont", 9),
        ).pack(side=tk.LEFT)
        self._skip_spin = tk.Spinbox(
            skip_row, from_=0, to=10000, increment=1, width=8,
            textvariable=self._skip_var, bg="#2d2d2d", fg="#d4d4d4",
            insertbackground="#d4d4d4", buttonbackground="#3c3c3c",
            relief=tk.FLAT, font=("TkDefaultFont", 10),
        )
        self._skip_spin.pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(
            skip_frame,
            text="0 keeps every frame. 6 keeps 1 frame then skips 6 (every 7th).",
            bg="#1e1e1e", fg="#888888", font=("TkDefaultFont", 8), anchor=tk.W,
        ).pack(fill=tk.X, padx=6, pady=(0, 4))

        cap_frame = tk.LabelFrame(
            self, text=" Limit Extracted Frames ",
            bg="#1e1e1e", fg="#aaaaaa", font=("TkDefaultFont", 8),
        )
        cap_frame.pack(fill=tk.X, **pad)
        cap_row = tk.Frame(cap_frame, bg="#1e1e1e")
        cap_row.pack(fill=tk.X, padx=6, pady=6)
        self._cap_chk = tk.Checkbutton(
            cap_row, text="Cap at",
            variable=self._cap_enabled_var, command=self._on_inputs_changed,
            bg="#1e1e1e", fg="#d4d4d4", selectcolor="#2d2d2d",
            activebackground="#1e1e1e", activeforeground="#d4d4d4",
            font=("TkDefaultFont", 9),
        )
        self._cap_chk.pack(side=tk.LEFT)
        self._cap_entry = tk.Entry(
            cap_row, textvariable=self._cap_var, width=10,
            bg="#2d2d2d", fg="#d4d4d4", insertbackground="#d4d4d4",
            relief=tk.FLAT, font=("TkDefaultFont", 10),
        )
        self._cap_entry.pack(side=tk.LEFT, padx=(6, 4))
        tk.Label(
            cap_row, text="frames (sampled evenly across the video)",
            bg="#1e1e1e", fg="#888888", font=("TkDefaultFont", 8),
        ).pack(side=tk.LEFT)

        self._extract_count_lbl = tk.Label(
            self, text="Will extract: —",
            bg="#1e1e1e", fg="#44dd44", font=("TkDefaultFont", 10, "bold"),
        )
        self._extract_count_lbl.pack(padx=14, pady=(2, 0), anchor=tk.W)

        out_frame = tk.LabelFrame(
            self, text=" Output Folder ",
            bg="#1e1e1e", fg="#aaaaaa", font=("TkDefaultFont", 8),
        )
        out_frame.pack(fill=tk.X, **pad)
        out_row = tk.Frame(out_frame, bg="#1e1e1e")
        out_row.pack(fill=tk.X, padx=6, pady=6)
        tk.Entry(
            out_row, textvariable=self._output_var, state="readonly",
            bg="#2d2d2d", fg="#d4d4d4", readonlybackground="#2d2d2d",
            width=48, relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self._browse_out_btn = tk.Button(
            out_row, text="Browse…", command=self._browse_output,
            bg="#3c3c3c", fg="#d4d4d4", relief=tk.FLAT,
            activebackground="#505050", cursor="hand2",
        )
        self._browse_out_btn.pack(side=tk.LEFT)

        self._progress_bar = ttk.Progressbar(
            self, mode="determinate", length=300,
        )
        self._progress_msg = tk.Label(
            self, text="", bg="#1e1e1e", fg="#888888",
            font=("TkDefaultFont", 8),
        )

        btn_row = tk.Frame(self, bg="#1e1e1e")
        btn_row.pack(fill=tk.X, padx=12, pady=(8, 12))
        self._extract_btn = tk.Button(
            btn_row, text="Extract", command=self._on_extract,
            bg="#0e639c", fg="white", relief=tk.FLAT,
            activebackground="#1177bb", width=10,
            cursor="hand2", state=tk.DISABLED,
        )
        self._extract_btn.pack(side=tk.RIGHT, padx=(6, 0))
        self._cancel_btn = tk.Button(
            btn_row, text="Cancel", command=self._on_cancel,
            bg="#3c3c3c", fg="#d4d4d4", relief=tk.FLAT,
            activebackground="#505050", width=10, cursor="hand2",
        )
        self._cancel_btn.pack(side=tk.RIGHT)

    def _browse_video(self):
        exts = " ".join(f"*{e}" for e in FRAME_EXTRACT_VIDEO_EXTS)
        path = filedialog.askopenfilename(
            parent=self,
            title="Select Video",
            initialdir=Path.home(),
            filetypes=[("Video files", exts), ("All files", "*.*")],
        )
        if not path:
            return
        self._video_var.set(path)
        self._video_path = Path(path)
        self._probe()

    def _browse_output(self):
        current = self._output_var.get() or str(IMAGES_DIR)
        path = filedialog.askdirectory(
            parent=self,
            title="Select Output Folder",
            initialdir=current if Path(current).exists() else str(Path.home()),
            mustexist=False,
        )
        if path:
            self._output_var.set(path)

    def _probe(self):
        from auto_annotator.frame_extractor import probe_video

        self._resolution_lbl.config(text="Resolution: —")
        self._total_lbl.config(text="Total frames: —")
        self._fps_lbl.config(text="FPS: —    Duration: —")
        self._status_lbl.config(text="Probing video…", fg="#888888")
        self._extract_btn.config(state=tk.DISABLED)
        self.update_idletasks()

        try:
            info = probe_video(self._video_path)
        except RuntimeError as e:
            self._video_info = None
            self._status_lbl.config(text=str(e), fg="#ff4444")
            self._extract_count_lbl.config(text="Will extract: —")
            return

        self._video_info = info
        self._resolution_lbl.config(text=f"Resolution: {info.resolution}")
        self._total_lbl.config(text=f"Total frames: {info.total_frames:,}")
        duration_str = self._format_duration(info.duration_sec)
        self._fps_lbl.config(
            text=f"FPS: {info.fps:.2f}    Duration: {duration_str}"
        )
        self._status_lbl.config(text="Ready to extract", fg="#44dd44")
        self._on_inputs_changed()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds <= 0:
            return "—"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"

    def _on_inputs_changed(self, *_args):
        if self._video_info is None:
            self._extract_count_lbl.config(text="Will extract: —")
            self._extract_btn.config(state=tk.DISABLED)
            return

        from auto_annotator.frame_extractor import compute_frame_indices

        skip = self._safe_int(self._skip_var, 0)
        cap = None
        if self._cap_enabled_var.get():
            cap = self._safe_int(self._cap_var, 0)
            if cap <= 0:
                cap = None

        self._indices = compute_frame_indices(
            self._video_info.total_frames, skip, cap
        )
        n = len(self._indices)
        self._extract_count_lbl.config(text=f"Will extract: {n:,} frames")
        self._extract_btn.config(
            state=tk.NORMAL if n > 0 else tk.DISABLED,
        )

    @staticmethod
    def _safe_int(var: tk.StringVar, default: int) -> int:
        try:
            return int(var.get())
        except (ValueError, tk.TclError):
            return default

    def _on_extract(self):
        if self._video_info is None or not self._indices:
            return

        output_dir = Path(self._output_var.get())
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror(
                "Output Folder Error",
                f"Could not create output folder:\n{e}",
                parent=self,
            )
            return

        self._browse_vid_btn.config(state=tk.DISABLED)
        self._browse_out_btn.config(state=tk.DISABLED)
        self._skip_spin.config(state=tk.DISABLED)
        self._cap_chk.config(state=tk.DISABLED)
        self._cap_entry.config(state=tk.DISABLED)
        self._extract_btn.config(state=tk.DISABLED)

        total = len(self._indices)
        self._progress_bar.config(maximum=total, value=0)
        self._progress_bar.pack(padx=12, pady=(4, 0))
        self._progress_msg.config(text=f"Extracting 0 / {total}")
        self._progress_msg.pack(padx=12, pady=(2, 0))

        self._cancel_event.clear()
        self._worker = threading.Thread(
            target=self._extract_worker,
            args=(self._video_path, output_dir, list(self._indices)),
            daemon=True,
        )
        self._worker.start()
        self._poll_id = self.after(FRAME_EXTRACT_POLL_MS, self._poll_extract)

    def _extract_worker(
        self, video_path: Path, output_dir: Path, indices: List[int]
    ):
        from auto_annotator.frame_extractor import extract_frames

        try:
            count = extract_frames(
                video_path,
                output_dir,
                indices,
                on_progress=lambda done, total: self._status_queue.put(
                    ("progress", (done, total))
                ),
                on_status=lambda msg: self._status_queue.put(("status", msg)),
                should_cancel=self._cancel_event.is_set,
            )
            if self._cancel_event.is_set():
                self._status_queue.put(("cancelled", count))
            else:
                self._status_queue.put(("done", count))
        except Exception as e:
            self._status_queue.put(("error", str(e)))

    def _poll_extract(self):
        if not self.winfo_exists():
            return
        while True:
            try:
                kind, payload = self._status_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "progress":
                done, total = payload
                self._progress_bar.config(value=done)
                self._progress_msg.config(text=f"Extracting {done} / {total}")
            elif kind == "status":
                self._status_lbl.config(text=payload, fg="#888888")
            elif kind == "done":
                self._progress_bar.pack_forget()
                self._progress_msg.config(text="")
                out_dir = self._output_var.get()
                self._status_lbl.config(
                    text=f"Extracted {payload} frames to {out_dir}",
                    fg="#44dd44",
                )
                self._extract_btn.config(
                    text="Done", state=tk.NORMAL, command=self._on_done,
                )
                return
            elif kind == "cancelled":
                self._progress_bar.pack_forget()
                self._progress_msg.config(text="")
                self._status_lbl.config(
                    text=f"Stopped after {payload} frames", fg="#ffaa00",
                )
                self._reenable_inputs()
                return
            elif kind == "error":
                self._progress_bar.pack_forget()
                self._progress_msg.config(text="")
                self._status_lbl.config(text=payload, fg="#ff4444")
                self._reenable_inputs()
                return

        if self._worker and self._worker.is_alive():
            self._poll_id = self.after(FRAME_EXTRACT_POLL_MS, self._poll_extract)

    def _reenable_inputs(self):
        self._browse_vid_btn.config(state=tk.NORMAL)
        self._browse_out_btn.config(state=tk.NORMAL)
        self._skip_spin.config(state=tk.NORMAL)
        self._cap_chk.config(state=tk.NORMAL)
        self._cap_entry.config(state=tk.NORMAL)
        self._on_inputs_changed()

    def _on_done(self):
        self.destroy()

    def _on_cancel(self):
        if self._worker and self._worker.is_alive():
            self._cancel_event.set()
            self._status_lbl.config(text="Cancelling…", fg="#ffaa00")
            return
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        self.destroy()

    def _on_cancel(self):
        self.confirmed = False
        self.destroy()


class EmptyStateDialog(tk.Toplevel):
    """Friendly screen shown when IMAGES_DIR is empty on startup."""

    OUTCOMES = ("retry", "extract", "cancel")

    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.title("No Images Found")
        self.resizable(False, False)
        self.configure(bg="#1e1e1e")
        self.grab_set()

        self.outcome: Optional[str] = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.update_idletasks()
        pw = parent.winfo_screenwidth()
        ph = parent.winfo_screenheight()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry(f"+{(pw - w) // 2}+{(ph - h) // 2}")

        self.wait_window(self)

    def _build_ui(self):
        tk.Label(
            self, text="No Images to Annotate", bg="#007acc", fg="white",
            font=("TkDefaultFont", 12, "bold"), pady=10,
        ).pack(fill=tk.X)
        tk.Label(
            self,
            text="Before you can annotate, drop some image files into:",
            bg="#1e1e1e", fg="#d4d4d4", font=("TkDefaultFont", 10),
        ).pack(fill=tk.X, padx=16, pady=(12, 4))
        tk.Label(
            self, text=str(IMAGES_DIR),
            bg="#2d2d2d", fg="#4fc1ff", font=("TkFixedFont", 9),
            padx=8, pady=6, anchor="w",
        ).pack(fill=tk.X, padx=16)
        tk.Label(
            self,
            text=("Supported: .jpg / .jpeg / .png\n"
                  "Don't have images yet? Extract frames from a video instead."),
            bg="#1e1e1e", fg="#888888", font=("TkDefaultFont", 9),
            justify="left",
        ).pack(fill=tk.X, padx=16, pady=(8, 4))

        btn_row = tk.Frame(self, bg="#1e1e1e")
        btn_row.pack(fill=tk.X, padx=16, pady=(8, 14))

        open_btn = tk.Button(
            btn_row, text="Open Folder",
            command=self._on_open_folder,
            bg="#3c3c3c", fg="#d4d4d4", activebackground="#505050",
            relief=tk.FLAT, cursor="hand2", width=14,
        )
        open_btn.pack(side=tk.LEFT)
        Tooltip(open_btn, "Open the input_images folder in your file manager so you can drag images in.")

        extract_btn = tk.Button(
            btn_row, text="Extract Frames…",
            command=self._on_extract,
            bg="#3c3c3c", fg="#d4d4d4", activebackground="#505050",
            relief=tk.FLAT, cursor="hand2", width=16,
        )
        extract_btn.pack(side=tk.LEFT, padx=6)
        Tooltip(extract_btn, "Pick a video file and pull frames out of it. Frames are saved into input_images automatically.")

        retry_btn = tk.Button(
            btn_row, text="Retry",
            command=self._on_retry,
            bg="#0e639c", fg="white", activebackground="#1177bb",
            relief=tk.FLAT, cursor="hand2", width=10,
        )
        retry_btn.pack(side=tk.RIGHT)
        Tooltip(retry_btn, "Check the folder again. Press this after you've added images.")

        cancel_btn = tk.Button(
            btn_row, text="Quit",
            command=self._on_cancel,
            bg="#3c3c3c", fg="#d4d4d4", activebackground="#505050",
            relief=tk.FLAT, cursor="hand2", width=8,
        )
        cancel_btn.pack(side=tk.RIGHT, padx=(0, 6))

    def _on_open_folder(self):
        _open_in_file_manager(IMAGES_DIR)

    def _on_extract(self):
        self.outcome = "extract"
        self.destroy()

    def _on_retry(self):
        self.outcome = "retry"
        self.destroy()

    def _on_cancel(self):
        self.outcome = "cancel"
        self.destroy()


def _open_in_file_manager(path: Path) -> None:
    """Best-effort opening of a folder in the OS file manager."""
    import os
    import sys
    try:
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(path)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


class HelpDialog(tk.Toplevel):
    """Workflow + shortcuts reference, opened from the Help menu."""

    SHORTCUTS = [
        ("S",            "Switch to Select tool"),
        ("D",            "Switch to Draw tool"),
        ("Ctrl+Z",       "Undo last change"),
        ("Del / BkSp",   "Delete selected box"),
        ("↑ ↓ ← →",      "Nudge selected box by 1 pixel"),
        ("← / →",        "(no selection) Previous / Next image"),
        ("1st letter",   "Pick draw class by first letter of its name"),
        ("0",            "Reset zoom + fit window"),
        ("+  /  −",      "Zoom in / Zoom out (keyboard)"),
        ("Scroll wheel", "Zoom to cursor"),
        ("Middle drag",  "Pan the canvas"),
        ("Space + drag", "Pan the canvas (alternate)"),
        ("Shift (hold)", "Temporarily hide all boxes"),
    ]

    WORKFLOW = [
        ("1. Get images",
         "Drop .jpg / .png files into ~/Desktop/auto_annotate/input_images. "
         "No images yet? Use Tools → Frame Extraction… to turn a video into images."),
        ("2. Pick a model",
         "On startup, select your .onnx model file and the matching labels.txt."),
        ("3. Auto-detect",
         "Each image is detected automatically when you open it. "
         "Tools → Annotate All Images runs detection across every image at once."),
        ("4. Filter results",
         "On the right, use the Class Filter to hide classes you don't care about, "
         "or raise per-class confidence to drop noisy boxes."),
        ("5. Edit boxes",
         "Use Select [S] to click, drag, resize, or re-class a box. "
         "Use Draw [D] to draw new boxes — pick the class first."),
        ("6. Autosave",
         "Edits are saved automatically — a moment after you pause, when you switch images, "
         "and when you close the window. Crash or power-cut recovery is automatic on next launch."),
        ("7. Export",
         "File → Export YOLO / Export COCO. A split dialog lets you choose train/val/test ratios."),
    ]

    def __init__(self, parent: tk.Misc):
        super().__init__(parent)
        self.title("Help — Auto Annotator")
        self.configure(bg="#1e1e1e")
        self.resizable(False, False)
        self.transient(parent)
        try:
            self.grab_set()
        except tk.TclError:
            pass

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self.update_idletasks()
        try:
            pw = parent.winfo_rootx() + parent.winfo_width() // 2
            ph = parent.winfo_rooty() + parent.winfo_height() // 2
            self.geometry(f"+{pw - self.winfo_reqwidth() // 2}+{ph - self.winfo_reqheight() // 2}")
        except Exception:
            pass

    def _build_ui(self):
        tk.Label(
            self, text="Help & Shortcuts", bg="#007acc", fg="white",
            font=("TkDefaultFont", 11, "bold"), pady=8,
        ).pack(fill=tk.X)

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        wf = tk.Frame(nb, bg="#1e1e1e")
        nb.add(wf, text="  Workflow  ")
        for title, body in self.WORKFLOW:
            tk.Label(
                wf, text=title, bg="#1e1e1e", fg="#4fc1ff",
                font=("TkDefaultFont", 10, "bold"), anchor="w",
            ).pack(fill=tk.X, padx=10, pady=(8, 2))
            tk.Label(
                wf, text=body, bg="#1e1e1e", fg="#d4d4d4",
                font=("TkDefaultFont", 9), anchor="w", justify="left",
                wraplength=540,
            ).pack(fill=tk.X, padx=20)

        sc = tk.Frame(nb, bg="#1e1e1e")
        nb.add(sc, text="  Shortcuts  ")
        for key, desc in self.SHORTCUTS:
            row = tk.Frame(sc, bg="#1e1e1e")
            row.pack(fill=tk.X, padx=14, pady=2)
            tk.Label(
                row, text=key, bg="#1e1e1e", fg="#4fc1ff",
                font=("TkFixedFont", 9), width=16, anchor="w",
            ).pack(side=tk.LEFT)
            tk.Label(
                row, text=desc, bg="#1e1e1e", fg="#d4d4d4",
                font=("TkDefaultFont", 9), anchor="w",
            ).pack(side=tk.LEFT)

        btn_row = tk.Frame(self, bg="#1e1e1e")
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(
            btn_row, text="Close", command=self.destroy,
            bg="#3c3c3c", fg="#d4d4d4", activebackground="#505050",
            relief=tk.FLAT, cursor="hand2", width=10,
        ).pack(side=tk.RIGHT)
