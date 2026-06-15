import tkinter as tk
from typing import Callable, Optional, Union

from auto_annotator.config import TOOLTIP_DELAY_MS


TextSource = Union[str, Callable[[], str]]


class Tooltip:
    def __init__(self, widget: tk.Widget, text: TextSource, *, delay_ms: int = TOOLTIP_DELAY_MS):
        self._widget = widget
        self._text = text
        self._delay = delay_ms
        self._after_id: Optional[str] = None
        self._tip: Optional[tk.Toplevel] = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _resolve_text(self) -> str:
        try:
            return self._text() if callable(self._text) else self._text
        except Exception:
            return ""

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self._widget.after(self._delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self._widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        text = self._resolve_text()
        if not text or self._tip is not None:
            return
        try:
            x = self._widget.winfo_rootx() + 12
            y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        except tk.TclError:
            return
        tip = tk.Toplevel(self._widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tip.configure(bg="#3c3c3c")
        label = tk.Label(
            tip,
            text=text,
            bg="#3c3c3c",
            fg="#d4d4d4",
            font=("TkDefaultFont", 8),
            padx=6,
            pady=3,
            justify="left",
            wraplength=320,
        )
        label.pack()
        self._tip = tip

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def attach(widget: tk.Widget, text: TextSource) -> Tooltip:
    return Tooltip(widget, text)
