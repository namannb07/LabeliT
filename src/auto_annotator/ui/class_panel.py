import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional

from auto_annotator.config import (
    CLASS_COLORS,
    CONF_SPINBOX_INCREMENT,
    DEFAULT_CONF,
)
from auto_annotator.ui.tooltip import Tooltip


class ClassPanel:
    """Owns all class-related UI: draw-class shortcut buttons (bottom bar),
    class filter (checkboxes + per-class confidence spinboxes), and the
    selected-box class buttons (right panel).

    Does not own the tk.Variables (the app provides them so that other code
    can read filter / confidence values without going through this panel);
    does own the per-button widget references for highlight management.
    """

    def __init__(
        self,
        labels: List[str],
        class_enabled_vars: List[tk.BooleanVar],
        class_conf_vars: List[tk.DoubleVar],
        on_select_draw_class: Callable[[int], None],
        on_filter_change: Callable[[], None],
        on_box_class_select: Callable[[int], None],
    ):
        self.labels = labels
        self.class_enabled_vars = class_enabled_vars
        self.class_conf_vars = class_conf_vars
        self._on_select_draw_class = on_select_draw_class
        self._on_filter_change = on_filter_change
        self._on_box_class_select = on_box_class_select

        self._draw_class_buttons: List[tk.Button] = []
        self._box_class_buttons: List[tk.Button] = []

    # ── Draw-class shortcut buttons (bottom bar) ──────────────────────────

    def build_draw_buttons(self, parent: tk.Frame) -> tk.Canvas:
        """Build horizontal-scrolling class buttons. Returns the scroll canvas
        so the caller can put it in their layout (it's not packed yet)."""
        tk.Button(
            parent, text="◀", bg="#3c3c3c", fg="#d4d4d4",
            activebackground="#505050", font=("TkDefaultFont", 8),
            padx=2, pady=0, borderwidth=0, relief="flat",
            command=lambda: btn_canvas.xview_scroll(-80, "units"),
        ).pack(side=tk.LEFT, padx=(0, 1), pady=8)
        btn_canvas = tk.Canvas(parent, bg="#252526", highlightthickness=0, height=36)
        btn_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=6, padx=0)
        inner = tk.Frame(btn_canvas, bg="#252526")
        btn_canvas.create_window((0, 0), window=inner, anchor="nw")
        self._draw_class_buttons = []
        for i, label in enumerate(self.labels):
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            btn = tk.Button(
                inner, text=f"■ {label}",
                bg="#2d2d2d", fg=color,
                activebackground="#3a3a3a", activeforeground=color,
                font=("TkDefaultFont", 8), padx=6, pady=2,
                borderwidth=1, relief="flat",
                command=lambda idx=i: self._on_select_draw_class(idx),
            )
            btn.pack(side=tk.LEFT, padx=2)
            self._draw_class_buttons.append(btn)
            shortcut = label[0].lower() if label and label[0].isalpha() else None
            tip = f"Draw new boxes as '{label}'"
            if shortcut:
                tip += f"  (shortcut: {shortcut.upper()})"
            Tooltip(btn, tip)
        inner.update_idletasks()
        btn_canvas.config(scrollregion=btn_canvas.bbox("all"), xscrollincrement=1)
        btn_canvas.bind("<Button-4>", lambda e: btn_canvas.xview_scroll(-80, "units"))
        btn_canvas.bind("<Button-5>", lambda e: btn_canvas.xview_scroll(80, "units"))
        btn_canvas.bind(
            "<MouseWheel>",
            lambda e: btn_canvas.xview_scroll(-80 if e.delta > 0 else 80, "units"),
        )
        tk.Button(
            parent, text="▶", bg="#3c3c3c", fg="#d4d4d4",
            activebackground="#505050", font=("TkDefaultFont", 8),
            padx=2, pady=0, borderwidth=0, relief="flat",
            command=lambda: btn_canvas.xview_scroll(80, "units"),
        ).pack(side=tk.LEFT, padx=(1, 4), pady=8)
        return btn_canvas

    def refresh_draw_class_highlight(self, active_idx: int) -> None:
        for j, btn in enumerate(self._draw_class_buttons):
            if j == active_idx:
                btn.config(bg="#094771", fg="white", activebackground="#0a5a99")
            else:
                color = CLASS_COLORS[j % len(CLASS_COLORS)]
                btn.config(bg="#2d2d2d", fg=color, activebackground="#3a3a3a")

    # ── Class filter (checkboxes + per-class confidence) ──────────────────

    def build_filter(self, parent: tk.LabelFrame) -> None:
        sel_row = tk.Frame(parent, bg="#1e1e1e")
        sel_row.pack(fill=tk.X, pady=(0, 4))
        all_on = ttk.Button(sel_row, text="All On", command=self._select_all)
        all_on.pack(side=tk.LEFT, padx=(0, 4))
        Tooltip(all_on, "Show every class on the canvas.")
        all_off = ttk.Button(sel_row, text="All Off", command=self._deselect_all)
        all_off.pack(side=tk.LEFT)
        Tooltip(all_off, "Hide every class. Useful when you want to enable just one or two.")

        hdr = tk.Frame(parent, bg="#1e1e1e")
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Class", bg="#1e1e1e", fg="#888",
                 font=("TkDefaultFont", 7), anchor="w").pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Label(hdr, text="Conf", bg="#1e1e1e", fg="#888",
                 font=("TkDefaultFont", 7), width=5).pack(side=tk.RIGHT)

        filter_canvas = tk.Canvas(parent, bg="#1e1e1e", highlightthickness=0,
                                  height=min(200, len(self.labels) * 26 + 4))
        filter_scroll = tk.Scrollbar(parent, orient=tk.VERTICAL,
                                     command=filter_canvas.yview)
        filter_canvas.configure(yscrollcommand=filter_scroll.set)
        if len(self.labels) > 7:
            filter_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        filter_canvas.pack(fill=tk.X)

        inner = tk.Frame(filter_canvas, bg="#1e1e1e")
        filter_canvas.create_window((0, 0), window=inner, anchor="nw")

        for i, label in enumerate(self.labels):
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            row = tk.Frame(inner, bg="#1e1e1e")
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text="■", fg=color, bg="#1e1e1e",
                     font=("TkDefaultFont", 10)).pack(side=tk.LEFT)
            chk = ttk.Checkbutton(row, text=label, variable=self.class_enabled_vars[i],
                                  command=self._on_filter_change)
            chk.pack(side=tk.LEFT, expand=True, fill=tk.X)
            Tooltip(chk, f"Show or hide '{label}' boxes on the canvas.")
            sp = ttk.Spinbox(row, from_=0.01, to=0.99, increment=CONF_SPINBOX_INCREMENT,
                             textvariable=self.class_conf_vars[i],
                             width=5, format="%.2f")
            sp.pack(side=tk.RIGHT)
            sp.bind("<Return>",   lambda e, idx=i: self._validate_conf(idx))
            sp.bind("<FocusOut>", lambda e, idx=i: self._validate_conf(idx))
            Tooltip(sp, f"Minimum confidence for '{label}'. Boxes below this are hidden.")

        inner.update_idletasks()
        filter_canvas.config(scrollregion=filter_canvas.bbox("all"))

    def apply_global_conf(self, value: float) -> None:
        val = round(value, 2)
        for var in self.class_conf_vars:
            var.set(val)

    # ── Selected-box class buttons ────────────────────────────────────────

    def build_selected_box_buttons(self, parent: tk.LabelFrame) -> None:
        tk.Label(parent, text="Change Class:", bg="#1e1e1e", fg="#aaaaaa",
                 font=("TkDefaultFont", 8)).pack(anchor="w")
        h = min(120, len(self.labels) * 26 + 4)
        box_canvas = tk.Canvas(parent, bg="#1e1e1e", highlightthickness=0, height=h)
        box_scroll = tk.Scrollbar(parent, orient=tk.VERTICAL, command=box_canvas.yview)
        box_canvas.configure(yscrollcommand=box_scroll.set)
        if len(self.labels) > 4:
            box_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        box_canvas.pack(fill=tk.X, pady=(2, 4))
        inner = tk.Frame(box_canvas, bg="#1e1e1e")
        box_canvas.create_window((0, 0), window=inner, anchor="nw")
        self._box_class_buttons = []
        for i, label in enumerate(self.labels):
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            row = tk.Frame(inner, bg="#1e1e1e")
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text="■", fg=color, bg="#1e1e1e",
                     font=("TkDefaultFont", 9)).pack(side=tk.LEFT)
            btn = tk.Button(
                row, text=label, bg="#2d2d2d", fg="#d4d4d4",
                activebackground="#3a3a3a", activeforeground="white",
                font=("TkDefaultFont", 8), padx=4, pady=1,
                borderwidth=0, relief="flat", anchor="w",
                command=lambda idx=i: self._on_box_class_select(idx),
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._box_class_buttons.append(btn)
            Tooltip(btn, f"Change the selected box to class '{label}'. Select a box first (Select tool, click on it).")
        inner.update_idletasks()
        box_canvas.config(scrollregion=box_canvas.bbox("all"))

    def refresh_selected_highlight(self, active_idx: Optional[int]) -> None:
        for j, btn in enumerate(self._box_class_buttons):
            if active_idx is not None and j == active_idx:
                btn.config(bg="#094771", fg="white")
            else:
                btn.config(bg="#2d2d2d", fg="#d4d4d4")

    # ── Internals ─────────────────────────────────────────────────────────

    def _select_all(self) -> None:
        for var in self.class_enabled_vars:
            var.set(True)
        self._on_filter_change()

    def _deselect_all(self) -> None:
        for var in self.class_enabled_vars:
            var.set(False)
        self._on_filter_change()

    def _validate_conf(self, class_idx: int) -> None:
        var = self.class_conf_vars[class_idx]
        try:
            v = round(float(var.get()), 2)
        except (tk.TclError, ValueError):
            var.set(DEFAULT_CONF)
            return
        var.set(max(0.01, min(0.99, v)))
