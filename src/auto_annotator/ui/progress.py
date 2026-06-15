import tkinter as tk
from tkinter import ttk
from typing import Optional


class ProgressDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        title: str = "Working…",
        message: str = "",
        total: Optional[int] = None,
        cancellable: bool = True,
    ):
        super().__init__(parent)
        self.title(title)
        self.configure(bg="#1e1e1e")
        self.resizable(False, False)
        self.transient(parent)
        try:
            self.grab_set()
        except tk.TclError:
            pass

        self._cancel_requested = False
        self._total = total
        self._closed = False

        self._message_var = tk.StringVar(value=message)
        self._detail_var = tk.StringVar(value="")

        outer = tk.Frame(self, bg="#1e1e1e", padx=20, pady=16)
        outer.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            outer,
            textvariable=self._message_var,
            bg="#1e1e1e",
            fg="#d4d4d4",
            font=("TkDefaultFont", 10),
            anchor="w",
            width=44,
        ).pack(fill=tk.X)

        if total is not None and total > 0:
            self._bar = ttk.Progressbar(outer, mode="determinate", maximum=total, length=360)
        else:
            self._bar = ttk.Progressbar(outer, mode="indeterminate", length=360)
            self._bar.start(12)
        self._bar.pack(fill=tk.X, pady=(10, 6))

        tk.Label(
            outer,
            textvariable=self._detail_var,
            bg="#1e1e1e",
            fg="#888888",
            font=("TkDefaultFont", 8),
            anchor="w",
        ).pack(fill=tk.X)

        if cancellable:
            btn_row = tk.Frame(outer, bg="#1e1e1e")
            btn_row.pack(fill=tk.X, pady=(10, 0))
            self._cancel_btn = ttk.Button(btn_row, text="Cancel", command=self._on_cancel)
            self._cancel_btn.pack(side=tk.RIGHT)
        else:
            self._cancel_btn = None

        self.protocol("WM_DELETE_WINDOW", self._on_cancel if cancellable else (lambda: None))

        self.update_idletasks()
        try:
            px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
            py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
            self.geometry(f"+{max(px, 0)}+{max(py, 0)}")
        except Exception:
            pass

    def _on_cancel(self):
        self._cancel_requested = True
        if self._cancel_btn is not None:
            self._cancel_btn.config(state=tk.DISABLED, text="Cancelling…")

    @property
    def cancelled(self) -> bool:
        return self._cancel_requested

    def set_message(self, message: str) -> None:
        self._message_var.set(message)
        self._pump()

    def set_detail(self, detail: str) -> None:
        self._detail_var.set(detail)
        self._pump()

    def step(self, current: int, detail: Optional[str] = None) -> None:
        if self._total is not None:
            self._bar["value"] = min(current, self._total)
        if detail is not None:
            self._detail_var.set(detail)
        self._pump()

    def _pump(self) -> None:
        try:
            self.update_idletasks()
            self.update()
        except tk.TclError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if isinstance(self._bar, ttk.Progressbar):
                try:
                    self._bar.stop()
                except Exception:
                    pass
            self.grab_release()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass
