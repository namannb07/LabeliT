import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import List, Optional

from PIL import Image

from auto_annotator.config import (
    AUTOSAVE_CEILING_MS,
    AUTOSAVE_DEBOUNCE_MS,
    CANVAS_H,
    CANVAS_W,
    COCO_ANNOTATIONS_DIR,
    DATASETS_DIR,
    DEFAULT_CONF,
    IMAGES_DIR,
    NUDGE_PX,
    YOLO_ANNOTATIONS_DIR,
    ZOOM_STEP_KEYBOARD,
    ZOOM_STEP_SCROLL,
)
from auto_annotator.detector import TRTDetector
from auto_annotator.dialogs import (
    ExportEngineDialog,
    FrameExtractionDialog,
    HelpDialog,
    SplitConfigDialog,
    _open_in_file_manager,
)
from auto_annotator.models import BoundingBox
from auto_annotator.store import AnnotationStore, split_image_paths
from auto_annotator.ui.bbox_editor import BboxEditor
from auto_annotator.ui.box_view import BoxView
from auto_annotator.ui.canvas_renderer import HIDDEN, CanvasRenderer
from auto_annotator.ui.canvas_transform import CanvasTransform
from auto_annotator.ui.class_panel import ClassPanel
from auto_annotator.ui.dataset_viz import DatasetVisualizationDialog
from auto_annotator.ui.file_panel import FilePanel
from auto_annotator.ui.progress import ProgressDialog
from auto_annotator.ui.tooltip import Tooltip
from auto_annotator.ui.undo import UndoStack


class AnnotatorApp(tk.Tk):
    def __init__(self, detector: TRTDetector, store: AnnotationStore,
                 image_paths: List[Path], labels: List[str]):
        super().__init__()
        self.detector = detector
        self.store = store
        self.image_paths = image_paths
        self.labels = labels

        self.current_path: Optional[Path] = None
        self.current_boxes: List[BoxView] = []
        self._pil_image: Optional[Image.Image] = None

        self._autosave_after_id: Optional[str] = None

        self.transform = CanvasTransform()

        self._tool_mode_var = tk.StringVar(value="SELECT")
        self._draw_class_var = tk.IntVar(value=0)

        self._boxes_hidden: bool = False

        self.undo_stack = UndoStack()

        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        self.auto_apply_conf_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready.")
        self.box_count_var = tk.StringVar(value="0 boxes")
        self.tool_label_var = tk.StringVar(value="Tool: SELECT")
        self.image_index_var = tk.StringVar(value="Image — / —")
        self.zoom_status_var = tk.StringVar(value="100%")
        self.class_enabled_vars: List[tk.BooleanVar] = [tk.BooleanVar(value=True) for _ in self.labels]
        self.class_conf_vars: List[tk.DoubleVar] = [tk.DoubleVar(value=DEFAULT_CONF) for _ in self.labels]
        self._tool_buttons: dict = {}
        self._viz_dialog: Optional[DatasetVisualizationDialog] = None

        self.class_panel = ClassPanel(
            labels=self.labels,
            class_enabled_vars=self.class_enabled_vars,
            class_conf_vars=self.class_conf_vars,
            on_select_draw_class=self._select_draw_class,
            on_filter_change=self._on_class_filter_change,
            on_box_class_select=self._on_box_class_select,
        )

        self.file_panel = FilePanel(
            image_paths=self.image_paths,
            store=self.store,
            on_select=self._on_file_panel_select,
            get_current_path=lambda: self.current_path,
        )

        model_tag = detector.model_type.upper() if detector.model_type else "?"
        res_tag = f"{detector.net_w}×{detector.net_h}"
        self.title(f"Annotator — {model_tag} {res_tag} — {len(labels)} classes")
        self.geometry("1400x860")
        self.resizable(True, True)
        self.configure(bg="#1e1e1e")

        self._build_menu_bar()
        self._build_layout()
        self.file_panel.populate()

        self.bind("<Delete>",   self._on_delete)
        self.bind("<BackSpace>", self._on_delete)
        self.bind("<Control-z>", self._on_undo)
        self.bind("s", lambda e: self._set_tool("select"))
        self.bind("S", lambda e: self._set_tool("select"))
        self.bind("d", lambda e: self._set_tool("draw"))
        self.bind("D", lambda e: self._set_tool("draw"))
        self.bind("<Left>",  lambda e: self.editor.nudge(-NUDGE_PX, 0) if self.editor.get_selected() else self._navigate_image(-1))
        self.bind("<Right>", lambda e: self.editor.nudge(NUDGE_PX, 0) if self.editor.get_selected() else self._navigate_image(1))
        self.bind("<Up>",    lambda e: self.editor.nudge(0, -NUDGE_PX))
        self.bind("<Down>",  lambda e: self.editor.nudge(0, NUDGE_PX))
        self.bind("0", self._on_zoom_reset)
        self.bind("<plus>",  self._on_zoom_in_kbd)
        self.bind("<equal>", self._on_zoom_in_kbd)
        self.bind("<minus>", self._on_zoom_out_kbd)
        self.bind("<KeyPress-space>",   self._on_space_down)
        self.bind("<KeyRelease-space>", self._on_space_up)
        self.bind("<KeyPress-Shift_L>",   lambda e: self._set_boxes_hidden(True))
        self.bind("<KeyPress-Shift_R>",   lambda e: self._set_boxes_hidden(True))
        self.bind("<KeyRelease-Shift_L>", lambda e: self._set_boxes_hidden(False))
        self.bind("<KeyRelease-Shift_R>", lambda e: self._set_boxes_hidden(False))

        _reserved_keys = {'s', 'd', '0', '+', '=', '-', ' '}
        _registered_class_keys: set = set()
        for _i, _label in enumerate(self.labels):
            if not _label:
                continue
            _key = _label[0].lower()
            if _key.isalpha() and _key not in _reserved_keys and _key not in _registered_class_keys:
                self.bind(_key, lambda e, i=_i: self._select_draw_class(i))
                _registered_class_keys.add(_key)

        if self.labels:
            self._select_draw_class(0)

        if self.image_paths:
            self.file_panel.select_first()
        self._update_image_index()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(AUTOSAVE_CEILING_MS, self._autosave_ceiling_tick)

    # ── Layout builders ───────────────────────────────────────────────────────

    def _build_menu_bar(self):
        menubar = tk.Menu(self, bg="#252526", fg="#d4d4d4",
                          activebackground="#094771", activeforeground="white",
                          borderwidth=0)

        file_menu = tk.Menu(menubar, tearoff=0, bg="#252526", fg="#d4d4d4",
                            activebackground="#094771", activeforeground="white")
        file_menu.add_command(label="Open Images Folder…",
                              command=lambda: _open_in_file_manager(IMAGES_DIR))
        file_menu.add_command(label="Show Datasets Folder",
                              command=lambda: _open_in_file_manager(DATASETS_DIR))
        file_menu.add_separator()
        file_menu.add_command(label="Import Annotations…", command=self._on_import)
        file_menu.add_separator()
        file_menu.add_command(label="Export YOLO…", command=self._on_export_all)
        file_menu.add_command(label="Export COCO…", command=self._on_export_coco)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0, bg="#252526", fg="#d4d4d4",
                            activebackground="#094771", activeforeground="white")
        edit_menu.add_command(label="Undo", accelerator="Ctrl+Z",
                              command=lambda: self._on_undo(None))
        edit_menu.add_command(label="Delete Selected", accelerator="Del",
                              command=lambda: self._on_delete(None))
        menubar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = tk.Menu(menubar, tearoff=0, bg="#252526", fg="#d4d4d4",
                            activebackground="#094771", activeforeground="white")
        view_menu.add_command(label="Zoom In",  accelerator="+",
                              command=self._on_zoom_in_kbd)
        view_menu.add_command(label="Zoom Out", accelerator="−",
                              command=self._on_zoom_out_kbd)
        view_menu.add_command(label="Fit Window", accelerator="0",
                              command=self._on_zoom_reset)
        view_menu.add_separator()
        view_menu.add_command(label="Hide Boxes (hold Shift)", state=tk.DISABLED)
        menubar.add_cascade(label="View", menu=view_menu)

        tools_menu = tk.Menu(menubar, tearoff=0, bg="#252526", fg="#d4d4d4",
                             activebackground="#094771", activeforeground="white")
        tools_menu.add_command(label="Select Tool", accelerator="S",
                               command=lambda: self._set_tool("select"))
        tools_menu.add_command(label="Draw Tool", accelerator="D",
                               command=lambda: self._set_tool("draw"))
        tools_menu.add_separator()
        tools_menu.add_command(label="Re-run Inference on Current",
                               command=self._on_rerun)
        tools_menu.add_command(label="Annotate All Images…",
                               command=self._on_annotate_all)
        tools_menu.add_separator()
        tools_menu.add_command(label="Visualize Dataset…",
                               command=self._on_visualize_dataset)
        tools_menu.add_separator()
        tools_menu.add_command(label="Frame Extraction…",
                               command=lambda: FrameExtractionDialog(self))
        tools_menu.add_command(label="Build TensorRT Engine…",
                               command=lambda: ExportEngineDialog(self))
        menubar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menubar, tearoff=0, bg="#252526", fg="#d4d4d4",
                            activebackground="#094771", activeforeground="white")
        help_menu.add_command(label="Workflow & Shortcuts…", command=self._on_help)
        help_menu.add_command(label="About", command=self._on_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _on_visualize_dataset(self):
        if self._viz_dialog is not None and self._viz_dialog.winfo_exists():
            self._viz_dialog.lift()
            self._viz_dialog.focus_set()
            self._viz_dialog.refresh()
            return
        self._viz_dialog = DatasetVisualizationDialog(self)

    def _on_help(self):
        HelpDialog(self)

    def _on_about(self):
        messagebox.showinfo(
            "About",
            "TensorRT Auto-Annotator\n"
            "Bulk image annotation for Jetson / JetPack.\n\n"
            "Tip: Help → Workflow & Shortcuts opens a full reference.",
            parent=self,
        )

    def _build_layout(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#1e1e1e", foreground="#d4d4d4", fieldbackground="#2d2d2d")
        style.configure("TButton", background="#3c3c3c", foreground="#d4d4d4", borderwidth=1, padding=4)
        style.map("TButton", background=[("active", "#505050")])
        style.configure("Accent.TButton", background="#094771", foreground="#ffffff", borderwidth=1, padding=4)
        style.map("Accent.TButton", background=[("active", "#1177bb")])
        style.configure("TScale", background="#1e1e1e", troughcolor="#3c3c3c")
        style.configure("TCheckbutton", background="#1e1e1e", foreground="#d4d4d4")
        style.configure("TLabelframe", background="#1e1e1e", foreground="#d4d4d4", bordercolor="#555")
        style.configure("TLabelframe.Label", background="#1e1e1e", foreground="#aaaaaa")
        style.configure("TCombobox", fieldbackground="#2d2d2d", foreground="#d4d4d4", background="#3c3c3c")

        status_bar = tk.Frame(self, bg="#007acc", height=24)
        status_bar.pack(side=tk.TOP, fill=tk.X)
        status_bar.pack_propagate(False)
        tk.Label(status_bar, textvariable=self.status_var, bg="#007acc", fg="white",
                 anchor="w", padx=8, font=("TkDefaultFont", 9)).pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(status_bar, textvariable=self.zoom_status_var, bg="#007acc", fg="white",
                 anchor="e", padx=8, font=("TkDefaultFont", 9, "bold"), width=6
                 ).pack(side=tk.RIGHT)
        tk.Frame(status_bar, bg="#005a99", width=1).pack(side=tk.RIGHT, fill=tk.Y, pady=4)
        tk.Label(status_bar, textvariable=self.box_count_var, bg="#007acc", fg="white",
                 anchor="e", padx=8, font=("TkDefaultFont", 9)
                 ).pack(side=tk.RIGHT)
        tk.Frame(status_bar, bg="#005a99", width=1).pack(side=tk.RIGHT, fill=tk.Y, pady=4)
        tk.Label(status_bar, textvariable=self.tool_label_var, bg="#007acc", fg="white",
                 anchor="e", padx=8, font=("TkDefaultFont", 9, "bold")
                 ).pack(side=tk.RIGHT)
        tk.Frame(status_bar, bg="#005a99", width=1).pack(side=tk.RIGHT, fill=tk.Y, pady=4)
        tk.Label(status_bar, textvariable=self.image_index_var, bg="#007acc", fg="white",
                 anchor="e", padx=8, font=("TkDefaultFont", 9)
                 ).pack(side=tk.RIGHT)

        bottom = tk.Frame(self, bg="#252526", height=60)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        bottom.pack_propagate(False)

        tk.Label(bottom, text="Tool:", bg="#252526", fg="#888",
                 font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(8, 2), pady=8)
        select_btn = ttk.Button(bottom, text="Select [S]",
                                command=lambda: self._set_tool("select"))
        select_btn.pack(side=tk.LEFT, padx=2, pady=8)
        draw_btn = ttk.Button(bottom, text="Draw [D]",
                              command=lambda: self._set_tool("draw"))
        draw_btn.pack(side=tk.LEFT, padx=2, pady=8)
        self._tool_buttons = {"select": select_btn, "draw": draw_btn}
        Tooltip(select_btn, "Select tool — click a box to select, then drag to move, drag handles to resize, or click a class on the right to re-classify. Shortcut: S")
        Tooltip(draw_btn, "Draw tool — drag on the image to draw a new box. The class chosen above is used for new boxes. Shortcut: D")

        tk.Label(bottom, text="Class:", bg="#252526", fg="#888",
                 font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(6, 2), pady=8)
        self.class_panel.build_draw_buttons(bottom)

        tk.Frame(bottom, bg="#555", width=1).pack(side=tk.LEFT, fill=tk.Y, pady=10)

        import_btn = ttk.Button(bottom, text="Import",       command=self._on_import)
        import_btn.pack(side=tk.LEFT, padx=4, pady=8)
        Tooltip(import_btn, "Load existing YOLO or COCO annotations from disk and merge them into this session.")

        annotate_all_btn = ttk.Button(bottom, text="Annotate All",  command=self._on_annotate_all)
        annotate_all_btn.pack(side=tk.LEFT, padx=6, pady=8)
        Tooltip(annotate_all_btn, "Run detection on every image. Existing manual edits will be replaced with fresh detections.")

        export_yolo_btn = ttk.Button(bottom, text="Export YOLO",   command=self._on_export_all)
        export_yolo_btn.pack(side=tk.LEFT, padx=4, pady=8)
        Tooltip(export_yolo_btn, "Write a YOLO-format dataset (.txt labels + data.yaml). You'll be asked for a name and train/val/test split.")

        export_coco_btn = ttk.Button(bottom, text="Export COCO",   command=self._on_export_coco)
        export_coco_btn.pack(side=tk.LEFT, padx=4, pady=8)
        Tooltip(export_coco_btn, "Write a COCO-format dataset (annotations.json + images). You'll be asked for a name and train/val/test split.")

        viz_btn = ttk.Button(bottom, text="Visualize", command=self._on_visualize_dataset)
        viz_btn.pack(side=tk.LEFT, padx=4, pady=8)
        Tooltip(viz_btn, "Open a dataset visualization window with class distribution and dataset stats. Non-modal — you can keep annotating while it's open.")

        main = tk.Frame(self, bg="#1e1e1e")
        main.pack(fill=tk.BOTH, expand=True)

        self._build_left(main)
        self._build_right(main)
        self._build_center(main)

    def _build_left(self, parent):
        self.file_panel.build(parent)

    def _build_center(self, parent):
        center = tk.Frame(parent, bg="#1e1e1e")
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            center,
            bg="#2d2d2d",
            width=CANVAS_W,
            height=CANVAS_H,
            highlightthickness=0,
            cursor="arrow",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.renderer = CanvasRenderer(self.canvas, self.transform)
        self.editor = BboxEditor(
            canvas=self.canvas,
            transform=self.transform,
            undo_stack=self.undo_stack,
            get_current_path=lambda: self.current_path,
            get_current_boxes=lambda: self.current_boxes,
            get_class_enabled=self._is_class_enabled,
            get_draw_class=self._draw_class_var.get,
            on_changed=self._on_editor_changed,
            on_selection_changed=self._refresh_box_class_buttons,
        )

        self.canvas.bind("<Button-1>",        self.editor.on_click)
        self.canvas.bind("<B1-Motion>",       self.editor.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.editor.on_release)
        self.canvas.bind("<Button-2>",        self._on_pan_start)
        self.canvas.bind("<B2-Motion>",       self._on_pan_motion)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.canvas.bind("<Button-4>",        self._on_zoom_in)
        self.canvas.bind("<Button-5>",        self._on_zoom_out)
        self.canvas.bind("<MouseWheel>",      self._on_mousewheel)

    def _build_right(self, parent):
        right = tk.Frame(parent, bg="#252526", width=270)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        det_frame = ttk.LabelFrame(right, text="Confidence Threshold", padding=8)
        det_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(det_frame, text="Default for all classes (drag slider):", bg="#1e1e1e", fg="#aaaaaa",
                 font=("TkDefaultFont", 8)).pack(anchor="w")
        self.conf_label = tk.Label(det_frame, text=f"{DEFAULT_CONF:.2f}", bg="#1e1e1e",
                                   fg="#4fc1ff", font=("TkDefaultFont", 10, "bold"))
        self.conf_label.pack(anchor="w")
        conf_slider = ttk.Scale(det_frame, from_=0.05, to=0.95, orient=tk.HORIZONTAL,
                                variable=self.conf_var, command=self._on_conf_move)
        conf_slider.pack(fill=tk.X, pady=4)
        Tooltip(conf_slider, "Minimum confidence for displayed detections. Lower = more boxes; higher = fewer, more certain boxes.")

        auto_chk = ttk.Checkbutton(
            det_frame,
            text="Auto-apply to all classes as I drag",
            variable=self.auto_apply_conf_var,
        )
        auto_chk.pack(anchor="w", pady=(2, 4))
        Tooltip(auto_chk, "When on, moving the slider immediately updates every per-class confidence below. Turn off if you want to tune classes individually.")

        det_btn_row = tk.Frame(det_frame, bg="#1e1e1e")
        det_btn_row.pack(fill=tk.X, pady=(0, 2))
        apply_btn = ttk.Button(det_btn_row, text="Apply to All",
                               command=self._on_apply_global_conf)
        apply_btn.pack(side=tk.LEFT, padx=(0, 4))
        Tooltip(apply_btn, "Copy the slider value into every per-class confidence spinbox below.")
        rerun_btn = ttk.Button(det_btn_row, text="Re-run Inference",
                               command=self._on_rerun)
        rerun_btn.pack(side=tk.LEFT)
        Tooltip(rerun_btn, "Re-run detection on the current image only, using current thresholds.")

        view_frame = ttk.LabelFrame(right, text="View", padding=8)
        view_frame.pack(fill=tk.X, padx=8, pady=4)
        zoom_row = tk.Frame(view_frame, bg="#1e1e1e")
        zoom_row.pack(fill=tk.X)
        zoom_out_btn = ttk.Button(zoom_row, text="−", width=3,
                                  command=lambda: self._zoom_at(
                                      (self.canvas.winfo_width() or CANVAS_W) / 2,
                                      (self.canvas.winfo_height() or CANVAS_H) / 2, 1 / ZOOM_STEP_KEYBOARD))
        zoom_out_btn.pack(side=tk.LEFT)
        Tooltip(zoom_out_btn, "Zoom out (or press −). Use mouse scroll-wheel to zoom around the cursor.")

        fit_btn = ttk.Button(zoom_row, text="Fit [0]", command=self._on_zoom_reset)
        fit_btn.pack(side=tk.LEFT, padx=4)
        Tooltip(fit_btn, "Fit the whole image to the window (press 0).")

        zoom_in_btn = ttk.Button(zoom_row, text="+", width=3,
                                 command=lambda: self._zoom_at(
                                     (self.canvas.winfo_width() or CANVAS_W) / 2,
                                     (self.canvas.winfo_height() or CANVAS_H) / 2, ZOOM_STEP_KEYBOARD))
        zoom_in_btn.pack(side=tk.LEFT)
        Tooltip(zoom_in_btn, "Zoom in (or press +). Use mouse scroll-wheel to zoom around the cursor.")

        self._zoom_label = tk.Label(view_frame, text="100%", bg="#1e1e1e", fg="#888",
                                    font=("TkDefaultFont", 8))
        self._zoom_label.pack(anchor="w", pady=(4, 0))

        filter_frame = ttk.LabelFrame(right, text="Class Filter", padding=8)
        filter_frame.pack(fill=tk.X, padx=8, pady=4)
        self.class_panel.build_filter(filter_frame)

        sel_frame = ttk.LabelFrame(right, text="Selected Box", padding=8)
        sel_frame.pack(fill=tk.X, padx=8, pady=4)
        self.class_panel.build_selected_box_buttons(sel_frame)
        del_btn = ttk.Button(sel_frame, text="Delete Selected  [Del]",
                             command=lambda: self._on_delete(None))
        del_btn.pack(fill=tk.X)
        Tooltip(del_btn, "Delete the currently selected box. Shortcut: Del or Backspace.")

        kb_frame = ttk.LabelFrame(right, text="Shortcuts", padding=8)
        kb_frame.pack(fill=tk.X, padx=8, pady=4)
        for key, desc in [
            ("S",            "Select tool"),
            ("D",            "Draw tool"),
            ("Ctrl+Z",       "Undo"),
            ("Del",          "Delete selected"),
            ("↑/↓/←/→",      "Nudge box 1px"),
            ("←/→ (no sel)", "Prev/Next image"),
            ("1st letter",   "Select class"),
            ("Scroll",       "Zoom to cursor"),
            ("Mid-drag",     "Pan"),
            ("Space+drag",   "Pan"),
            ("Shift (hold)", "Hide all boxes"),
            ("0",            "Fit window"),
            ("+/-",          "Zoom in/out"),
        ]:
            row = tk.Frame(kb_frame, bg="#1e1e1e")
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=key, bg="#1e1e1e", fg="#4fc1ff",
                     font=("TkFixedFont", 8), width=12, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=desc, bg="#1e1e1e", fg="#888",
                     font=("TkDefaultFont", 8)).pack(side=tk.LEFT)
        help_btn = ttk.Button(kb_frame, text="Open Help…", command=self._on_help)
        help_btn.pack(fill=tk.X, pady=(6, 0))
        Tooltip(help_btn, "Open the full Help dialog (workflow walkthrough + every shortcut).")

    def _update_file_list_colors(self):
        self.file_panel.refresh_colors()

    # ── Canvas helpers ────────────────────────────────────────────────────────

    def _set_boxes_hidden(self, hidden: bool):
        if self._boxes_hidden != hidden:
            self._boxes_hidden = hidden
            self._redraw_canvas()

    def _domain_boxes(self) -> List[BoundingBox]:
        return [bv.box for bv in self.current_boxes]

    def _persist_current(self) -> None:
        if self.current_path:
            self.store.set_boxes(self.current_path, self._domain_boxes())
            self._schedule_autosave()

    # ── Autosave ──────────────────────────────────────────────────────────────

    def _schedule_autosave(self) -> None:
        if self._autosave_after_id is not None:
            self.after_cancel(self._autosave_after_id)
        self._autosave_after_id = self.after(
            AUTOSAVE_DEBOUNCE_MS, self._autosave_flush
        )

    def _autosave_flush(self) -> None:
        self._autosave_after_id = None
        try:
            n = self.store.save_dirty()
        except Exception as e:
            print(f"[autosave] flush failed: {e}", file=sys.stderr)
            return
        if n:
            self._update_file_list_colors()
            self._update_status(f"Autosaved {n} image{'s' if n != 1 else ''}")

    def _autosave_ceiling_tick(self) -> None:
        try:
            self.store.save_dirty()
        except Exception as e:
            print(f"[autosave] ceiling flush failed: {e}", file=sys.stderr)
        finally:
            self.after(AUTOSAVE_CEILING_MS, self._autosave_ceiling_tick)

    def _on_close(self) -> None:
        if self._autosave_after_id is not None:
            try:
                self.after_cancel(self._autosave_after_id)
            except Exception:
                pass
            self._autosave_after_id = None
        try:
            self.store.save_dirty()
        except Exception as e:
            print(f"[autosave] close-flush failed: {e}", file=sys.stderr)
        self.destroy()

    def _is_class_enabled(self, class_id: int) -> bool:
        if class_id < 0 or class_id >= len(self.class_enabled_vars):
            return False
        return self.class_enabled_vars[class_id].get()

    def _on_editor_changed(self, committed: bool) -> None:
        if committed:
            self._persist_current()
            self._update_file_list_colors()
        self._redraw_canvas()

    def _redraw_canvas(self):
        class_enabled = [v.get() for v in self.class_enabled_vars]
        visible = self.renderer.render(
            self._pil_image, self.current_boxes, self._boxes_hidden,
            self.labels, class_enabled,
        )
        if visible == HIDDEN:
            self.box_count_var.set("boxes hidden")
        else:
            self.box_count_var.set(f"{visible} box{'es' if visible != 1 else ''}")
        zoom_pct = f"{int(self.transform.zoom * 100)}%"
        self._zoom_label.config(text=zoom_pct)
        self.zoom_status_var.set(zoom_pct)

    # ── Zoom / pan ────────────────────────────────────────────────────────────

    def _zoom_at(self, canvas_x: float, canvas_y: float, factor: float):
        if self._pil_image is None:
            return
        cw = self.canvas.winfo_width() or CANVAS_W
        ch = self.canvas.winfo_height() or CANVAS_H
        self.transform.zoom_at(
            canvas_x, canvas_y, factor,
            cw, ch, self._pil_image.width, self._pil_image.height,
        )
        self._redraw_canvas()

    def _on_zoom_in(self, event):
        self._zoom_at(event.x, event.y, ZOOM_STEP_SCROLL)

    def _on_zoom_out(self, event):
        self._zoom_at(event.x, event.y, 1 / ZOOM_STEP_SCROLL)

    def _on_mousewheel(self, event):
        factor = ZOOM_STEP_SCROLL if event.delta > 0 else 1 / ZOOM_STEP_SCROLL
        self._zoom_at(event.x, event.y, factor)

    def _on_zoom_reset(self, event=None):
        self.transform.reset_view()
        self._redraw_canvas()

    def _on_zoom_in_kbd(self, event=None):
        cw = self.canvas.winfo_width() or CANVAS_W
        ch = self.canvas.winfo_height() or CANVAS_H
        self._zoom_at(cw / 2, ch / 2, ZOOM_STEP_KEYBOARD)

    def _on_zoom_out_kbd(self, event=None):
        cw = self.canvas.winfo_width() or CANVAS_W
        ch = self.canvas.winfo_height() or CANVAS_H
        self._zoom_at(cw / 2, ch / 2, 1 / ZOOM_STEP_KEYBOARD)

    def _on_pan_start(self, event):
        self.transform.start_pan_drag(float(event.x), float(event.y))
        self.canvas.config(cursor="fleur")

    def _on_pan_motion(self, event):
        if self.transform.update_pan_drag(event.x, event.y):
            self._redraw_canvas()

    def _on_pan_end(self, event):
        self.transform.end_pan_drag()
        self.canvas.config(cursor="crosshair" if self.editor.tool_mode == "draw" else "arrow")

    def _on_space_down(self, event):
        if isinstance(self.focus_get(), (ttk.Combobox, tk.Entry)):
            return
        self.transform.set_space_held(True)
        self.canvas.config(cursor="fleur")

    def _on_space_up(self, event):
        self.transform.set_space_held(False)
        self.canvas.config(cursor="crosshair" if self.editor.tool_mode == "draw" else "arrow")

    # ── Tool mode ─────────────────────────────────────────────────────────────

    def _set_tool(self, mode: str):
        self._tool_mode_var.set(mode.upper())
        self.tool_label_var.set(f"Tool: {mode.upper()}")
        self.editor.set_tool(mode)
        for k, btn in self._tool_buttons.items():
            btn.configure(style="Accent.TButton" if k == mode else "TButton")

    # ── Undo ──────────────────────────────────────────────────────────────────

    def _push_undo(self):
        self.undo_stack.push(self.current_path, self._domain_boxes())

    def _on_undo(self, event=None):
        if not self.current_path:
            return
        restored = self.undo_stack.pop(self.current_path)
        if restored is None:
            self._update_status("Nothing to undo.")
            return
        self.current_boxes = [BoxView(box=b) for b in restored]
        self.store.set_boxes(self.current_path, restored)
        self._redraw_canvas()
        self._update_status(f"Undo — {len(self.current_boxes)} box{'es' if len(self.current_boxes) != 1 else ''}")

    # ── Image navigation ──────────────────────────────────────────────────────

    def _navigate_image(self, delta: int):
        self.file_panel.navigate(delta)
        self._update_image_index()

    # ── Class button helpers ──────────────────────────────────────────────────

    def _select_draw_class(self, i: int):
        self._draw_class_var.set(i)
        self.class_panel.refresh_draw_class_highlight(i)

    def _refresh_box_class_buttons(self, active_idx: Optional[int]):
        self.class_panel.refresh_selected_highlight(active_idx)

    def _on_box_class_select(self, i: int):
        sel = self.editor.get_selected()
        if sel is None:
            return
        self._push_undo()
        sel.box.class_id = i
        self._persist_current()
        self._refresh_box_class_buttons(i)
        self._redraw_canvas()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_file_panel_select(self, path: Path) -> None:
        if self.current_path == path:
            return
        self.current_path = path
        self._update_image_index()
        self._load_image(path)

    def _update_image_index(self) -> None:
        try:
            idx = self.image_paths.index(self.current_path) + 1 if self.current_path else 0
        except ValueError:
            idx = 0
        total = len(self.image_paths)
        if idx and total:
            self.image_index_var.set(f"Image {idx} / {total}")
        else:
            self.image_index_var.set(f"Image — / {total}")

    def _load_image(self, path: Path):
        if self._autosave_after_id is not None:
            try:
                self.after_cancel(self._autosave_after_id)
            except Exception:
                pass
            self._autosave_after_id = None
        try:
            self.store.save_dirty()
        except Exception as e:
            print(f"[autosave] switch-flush failed: {e}", file=sys.stderr)

        self.transform.reset_view()
        self.editor.reset_drag_state()

        try:
            self._pil_image = Image.open(path).convert("RGB")
        except Exception as e:
            self._update_status(f"Error loading {path.name}: {e}")
            return

        if self.store.is_annotated(path):
            domain_boxes = list(self.store.get_boxes(path))
        else:
            domain_boxes = self._run_inference(path)
        self.current_boxes = [BoxView(box=b) for b in domain_boxes]

        self._redraw_canvas()
        self._update_file_list_colors()
        n = len(self.current_boxes)
        self._update_status(
            f"{path.name}  ({self._pil_image.width}×{self._pil_image.height})  — "
            f"{n} detection{'s' if n != 1 else ''}"
        )

    def _filter_boxes(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        result = []
        for box in boxes:
            cid = box.class_id
            if cid >= len(self.class_enabled_vars):
                continue
            if not self.class_enabled_vars[cid].get():
                continue
            if box.confidence < self.class_conf_vars[cid].get():
                continue
            result.append(box)
        return result

    def _run_inference(self, path: Path) -> List[BoundingBox]:
        min_conf = round(min(v.get() for v in self.class_conf_vars), 2)
        try:
            raw_boxes = self.detector.run(path, min_conf)
        except Exception as e:
            messagebox.showerror("Inference Error", str(e))
            raw_boxes = []
        boxes = self._filter_boxes(raw_boxes)
        self.store.set_boxes(path, boxes)
        self._update_file_list_colors()
        return boxes

    def _on_delete(self, event):
        self.editor.delete_selected()

    def _on_rerun(self):
        if self.current_path is None:
            return
        self.current_boxes = [BoxView(box=b) for b in self._run_inference(self.current_path)]
        self._redraw_canvas()
        n = len(self.current_boxes)
        self._update_status(
            f"Re-ran inference on {self.current_path.name} — {n} detection{'s' if n != 1 else ''}"
        )

    def _on_annotate_all(self):
        total = len(self.image_paths)
        if total == 0:
            return
        if not messagebox.askyesno(
            "Annotate All Images?",
            f"This will run detection on all {total} images and replace any existing\n"
            "annotations with fresh model output. Continue?",
            parent=self,
        ):
            return

        progress = ProgressDialog(
            self,
            title="Annotating All Images",
            message=f"Annotating 0 / {total}…",
            total=total,
            cancellable=True,
        )
        done = 0
        try:
            for i, path in enumerate(self.image_paths):
                if progress.cancelled:
                    break
                progress.set_message(f"Annotating {i + 1} / {total}: {path.name}")
                progress.step(i)
                min_conf = round(min(v.get() for v in self.class_conf_vars), 2)
                try:
                    raw_boxes = self.detector.run(path, min_conf)
                except Exception as e:
                    progress.close()
                    messagebox.showerror("Inference Error",
                                         f"Failed on {path.name}:\n{e}", parent=self)
                    return
                boxes = self._filter_boxes(raw_boxes)
                self.store.set_boxes(path, boxes)
                done = i + 1
            progress.step(done)
        finally:
            cancelled = progress.cancelled
            progress.close()

        self._update_file_list_colors()
        if self.current_path:
            self.current_boxes = [BoxView(box=b) for b in self.store.get_boxes(self.current_path)]
            self._redraw_canvas()
        if cancelled:
            self._update_status(f"Cancelled — annotated {done} / {total} images.")
        else:
            self._update_status(f"Annotated all {total} images. Review and save.")

    def _on_export_all(self):
        name = simpledialog.askstring(
            "YOLO Dataset Name",
            "Enter a name for this YOLO dataset\n(spaces become underscores):",
            parent=self,
        )
        if not name:
            return
        name = name.strip().replace(" ", "_")
        if not name:
            return

        annotated = [
            p for p in self.image_paths
            if self.store.is_annotated(p) and self.store.get_boxes(p)
        ]
        if not annotated:
            messagebox.showwarning(
                "No Annotations",
                "No images have annotations to export.",
                parent=self,
            )
            return

        dlg = SplitConfigDialog(self, len(annotated))
        if not dlg.confirmed:
            return

        out_dir = DATASETS_DIR / f"{name}_yolo"
        if out_dir.exists():
            if not messagebox.askyesno(
                "Overwrite?", f"'{name}_yolo' already exists.\nOverwrite it?", parent=self
            ):
                return
        try:
            splits = None
            if dlg.splits_enabled and not (
                dlg.train_pct == 100 and dlg.val_pct == 0 and dlg.test_pct == 0
            ):
                splits = split_image_paths(
                    annotated, dlg.train_pct, dlg.val_pct, dlg.test_pct, dlg.seed,
                )
            count = self.store.export_yolo(annotated, self.labels, out_dir, splits=splits)
            if splits:
                parts = " | ".join(
                    f"{k}: {len(v)}" for k, v in splits.items()
                )
                self._update_status(
                    f"YOLO export '{name}_yolo': {count} image{'s' if count != 1 else ''} "
                    f"({parts}) → {out_dir}/"
                )
            else:
                self._update_status(
                    f"YOLO export '{name}_yolo': {count} image{'s' if count != 1 else ''} → {out_dir}/"
                )
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _on_export_coco(self):
        name = simpledialog.askstring(
            "COCO Dataset Name",
            "Enter a name for this COCO dataset\n(spaces become underscores):",
            parent=self,
        )
        if not name:
            return
        name = name.strip().replace(" ", "_")
        if not name:
            return

        annotated = [
            p for p in self.image_paths
            if self.store.is_annotated(p) and self.store.get_boxes(p)
        ]
        if not annotated:
            messagebox.showwarning(
                "No Annotations",
                "No images have annotations to export.",
                parent=self,
            )
            return

        dlg = SplitConfigDialog(self, len(annotated))
        if not dlg.confirmed:
            return

        out_dir = DATASETS_DIR / f"{name}_coco"
        if out_dir.exists():
            if not messagebox.askyesno(
                "Overwrite?", f"'{name}_coco' already exists.\nOverwrite it?", parent=self
            ):
                return
        try:
            splits = None
            if dlg.splits_enabled and not (
                dlg.train_pct == 100 and dlg.val_pct == 0 and dlg.test_pct == 0
            ):
                splits = split_image_paths(
                    annotated, dlg.train_pct, dlg.val_pct, dlg.test_pct, dlg.seed,
                )
            n_imgs, n_anns = self.store.export_coco(
                annotated, self.labels, out_dir, splits=splits,
            )
            if splits:
                parts = " | ".join(
                    f"{k}: {len(v)}" for k, v in splits.items()
                )
                self._update_status(
                    f"COCO export '{name}_coco': {n_imgs} image{'s' if n_imgs != 1 else ''}, "
                    f"{n_anns} annotation{'s' if n_anns != 1 else ''} ({parts}) → {out_dir}/"
                )
            else:
                self._update_status(
                    f"COCO export '{name}_coco': {n_imgs} image{'s' if n_imgs != 1 else ''}, "
                    f"{n_anns} annotation{'s' if n_anns != 1 else ''} → {out_dir}/"
                )
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _on_import(self):
        dlg = tk.Toplevel(self)
        dlg.title("Import Annotations")
        dlg.configure(bg="#1e1e1e")
        dlg.resizable(False, False)
        dlg.grab_set()
        choice: List[Optional[str]] = [None]

        tk.Label(
            dlg, text="Select annotation format to import:",
            bg="#1e1e1e", fg="#d4d4d4", font=("TkDefaultFont", 10),
        ).pack(padx=20, pady=(12, 4))
        tk.Label(
            dlg, text=f"YOLO dir: {YOLO_ANNOTATIONS_DIR}\nCOCO dir: {COCO_ANNOTATIONS_DIR}",
            bg="#1e1e1e", fg="#888", font=("TkDefaultFont", 8), justify=tk.LEFT,
        ).pack(padx=20, pady=(0, 8))

        btn_row = tk.Frame(dlg, bg="#1e1e1e")
        btn_row.pack(padx=20, pady=(0, 12))

        def pick(fmt: str):
            choice[0] = fmt
            dlg.destroy()

        tk.Button(
            btn_row, text="YOLO", bg="#3c3c3c", fg="#d4d4d4",
            activebackground="#505050", padx=12, pady=4,
            command=lambda: pick("yolo"),
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            btn_row, text="COCO", bg="#3c3c3c", fg="#d4d4d4",
            activebackground="#505050", padx=12, pady=4,
            command=lambda: pick("coco"),
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            btn_row, text="Cancel", bg="#3c3c3c", fg="#d4d4d4",
            activebackground="#505050", padx=12, pady=4,
            command=dlg.destroy,
        ).pack(side=tk.LEFT, padx=4)

        dlg.wait_window(dlg)
        fmt = choice[0]
        if fmt is None:
            return

        try:
            if fmt == "yolo":
                n_imgs, n_boxes, skipped = self.store.import_yolo(
                    YOLO_ANNOTATIONS_DIR, self.image_paths, self.labels,
                )
            else:
                n_imgs, n_boxes, skipped = self.store.import_coco(
                    COCO_ANNOTATIONS_DIR, self.image_paths, self.labels,
                )
        except FileNotFoundError as e:
            messagebox.showerror("Import Error", str(e), parent=self)
            return
        except Exception as e:
            messagebox.showerror("Import Error", str(e), parent=self)
            return

        msg = (
            f"Imported {n_boxes} box{'es' if n_boxes != 1 else ''} "
            f"across {n_imgs} image{'s' if n_imgs != 1 else ''}."
        )
        if skipped:
            msg += (
                "\n\nSkipped classes not in current session:\n"
                + ", ".join(skipped)
            )
        messagebox.showinfo("Import Complete", msg, parent=self)
        self._update_status(f"Imported {n_boxes} boxes from {n_imgs} images ({fmt.upper()})")

        if self.current_path and self.store.is_annotated(self.current_path):
            self.current_boxes = [BoxView(box=b) for b in self.store.get_boxes(self.current_path)]
            self._redraw_canvas()
        self._update_file_list_colors()

    def _on_conf_move(self, val):
        v = round(float(val), 2)
        self.conf_label.config(text=f"{v:.2f}")
        if self.auto_apply_conf_var.get():
            self.class_panel.apply_global_conf(v)
            self._redraw_canvas()

    def _on_class_filter_change(self):
        self._redraw_canvas()

    def _on_apply_global_conf(self):
        val = round(self.conf_var.get(), 2)
        self.class_panel.apply_global_conf(val)
        self._redraw_canvas()
        self._update_status(f"Applied threshold {val:.2f} to all classes.")

    def _update_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()
