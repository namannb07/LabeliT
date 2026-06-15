import tkinter as tk
from pathlib import Path
from typing import Callable, List, Optional

from auto_annotator.store import AnnotationStore


class FilePanel:
    """Sidebar file list. Owns the tk.Listbox + scrollbar, populates from
    image_paths, color-codes rows by annotation state, and emits a callback
    when the user clicks an entry.

    Does not own current_path - the app does. The panel reads it via the
    get_current_path accessor for highlight rendering.
    """

    def __init__(
        self,
        image_paths: List[Path],
        store: AnnotationStore,
        on_select: Callable[[Path], None],
        get_current_path: Callable[[], Optional[Path]],
    ):
        self.image_paths = image_paths
        self.store = store
        self._on_select = on_select
        self._get_current_path = get_current_path
        self._listbox: Optional[tk.Listbox] = None
        self._hover_tip: Optional[tk.Toplevel] = None
        self._hover_idx: Optional[int] = None

    def build(self, parent: tk.Frame) -> None:
        left = tk.Frame(parent, bg="#252526", width=210)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(left, text="IMAGES", bg="#252526", fg="#aaaaaa",
                 font=("TkDefaultFont", 8), pady=6).pack(fill=tk.X, padx=8)

        list_frame = tk.Frame(left, bg="#252526")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        scrollbar = tk.Scrollbar(list_frame, bg="#3c3c3c")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            bg="#1e1e1e",
            fg="#d4d4d4",
            selectbackground="#094771",
            selectforeground="white",
            activestyle="none",
            font=("TkFixedFont", 8),
            borderwidth=0,
            highlightthickness=0,
        )
        self._listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._listbox.yview)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._listbox.bind("<Motion>", self._on_hover)
        self._listbox.bind("<Leave>",  lambda e: self._hide_hover())

    def populate(self) -> None:
        if self._listbox is None:
            return
        self._listbox.delete(0, tk.END)
        for p in self.image_paths:
            name = p.name
            display = name[:28] + ".." if len(name) > 30 else name
            self._listbox.insert(tk.END, display)
        self.refresh_colors()

    def refresh_colors(self) -> None:
        if self._listbox is None:
            return
        current = self._get_current_path()
        for i, p in enumerate(self.image_paths):
            if p == current:
                self._listbox.itemconfig(i, bg="#2d1a00", fg="#ffaa44")
            elif self.store.is_annotated(p):
                self._listbox.itemconfig(i, bg="#1a3a1a", fg="#6db96d")
            else:
                self._listbox.itemconfig(i, bg="#1e1e1e", fg="#d4d4d4")

    def select_first(self) -> None:
        if self._listbox is None or not self.image_paths:
            return
        self._listbox.selection_set(0)
        self._on_select(self.image_paths[0])

    def navigate(self, delta: int) -> None:
        if self._listbox is None or not self.image_paths:
            return
        current = self._get_current_path()
        if current is None:
            return
        try:
            idx = self.image_paths.index(current)
        except ValueError:
            return
        new_idx = max(0, min(len(self.image_paths) - 1, idx + delta))
        if new_idx == idx:
            return
        new_path = self.image_paths[new_idx]
        self._listbox.selection_clear(0, tk.END)
        self._listbox.selection_set(new_idx)
        self._listbox.see(new_idx)
        self._on_select(new_path)

    def _on_listbox_select(self, event) -> None:
        if self._listbox is None:
            return
        sel = self._listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self.image_paths[idx]
        self._on_select(path)

    def _on_hover(self, event) -> None:
        if self._listbox is None:
            return
        try:
            idx = self._listbox.nearest(event.y)
        except tk.TclError:
            return
        if idx < 0 or idx >= len(self.image_paths):
            self._hide_hover()
            return
        if idx == self._hover_idx and self._hover_tip is not None:
            return
        self._hover_idx = idx
        path = self.image_paths[idx]
        annotated = self.store.is_annotated(path)
        text = f"{path.name}\n{'✓ annotated' if annotated else '— not yet annotated'}"
        self._show_hover(text, event.x_root, event.y_root)

    def _show_hover(self, text: str, x_root: int, y_root: int) -> None:
        self._hide_hover()
        if self._listbox is None:
            return
        tip = tk.Toplevel(self._listbox)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x_root + 14}+{y_root + 10}")
        tip.configure(bg="#3c3c3c")
        tk.Label(
            tip, text=text, bg="#3c3c3c", fg="#d4d4d4",
            font=("TkDefaultFont", 8), padx=6, pady=3,
            justify="left",
        ).pack()
        self._hover_tip = tip

    def _hide_hover(self) -> None:
        self._hover_idx = None
        if self._hover_tip is not None:
            try:
                self._hover_tip.destroy()
            except Exception:
                pass
            self._hover_tip = None
