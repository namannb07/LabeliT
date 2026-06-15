import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import List

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from auto_annotator.config import CLASS_COLORS
from auto_annotator.store import AnnotationStore

BG = "#1e1e1e"
FG = "#d4d4d4"
PANEL_BG = "#252526"
MUTED = "#888888"


@dataclass
class DatasetStats:
    class_counts: List[int]
    total_images: int
    annotated_images: int
    total_boxes: int

    @property
    def unannotated_images(self) -> int:
        return self.total_images - self.annotated_images

    @property
    def avg_boxes_per_annotated(self) -> float:
        if self.annotated_images == 0:
            return 0.0
        return self.total_boxes / self.annotated_images

    @property
    def imbalance_ratio(self) -> float:
        nonzero = [c for c in self.class_counts if c > 0]
        if len(nonzero) < 2:
            return 1.0
        return max(nonzero) / min(nonzero)


def compute_dataset_stats(
    store: AnnotationStore,
    image_paths: List[Path],
    num_classes: int,
) -> DatasetStats:
    counts = [0] * num_classes
    annotated = 0
    total_boxes = 0
    path_set = set(image_paths)
    for path in image_paths:
        if not store.is_annotated(path):
            continue
        boxes = store.get_boxes(path)
        if not boxes:
            continue
        annotated += 1
        for b in boxes:
            total_boxes += 1
            if 0 <= b.class_id < num_classes:
                counts[b.class_id] += 1
    # Also include any annotated paths not in image_paths (defensive — shouldn't normally happen)
    for path, boxes in store._store.items():
        if path in path_set or not boxes:
            continue
        annotated += 1
        for b in boxes:
            total_boxes += 1
            if 0 <= b.class_id < num_classes:
                counts[b.class_id] += 1
    return DatasetStats(
        class_counts=counts,
        total_images=len(image_paths),
        annotated_images=annotated,
        total_boxes=total_boxes,
    )


class DatasetVisualizationDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent
        self.title("Dataset Visualization")
        self.configure(bg=BG)
        self.geometry("1080x600")
        self.transient(parent)
        # Intentionally NOT calling grab_set() — this window must be non-modal
        # so the user can keep annotating in the main window while it's open.

        self._fig: Figure | None = None
        self._canvas: FigureCanvasTkAgg | None = None
        self._summary_vars: dict[str, tk.StringVar] = {}

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh()

    def _build_layout(self):
        header = tk.Frame(self, bg=PANEL_BG, height=40)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text="Dataset Visualization",
            bg=PANEL_BG,
            fg=FG,
            font=("TkDefaultFont", 11, "bold"),
        ).pack(side=tk.LEFT, padx=12)
        ttk.Button(header, text="Close", command=self._on_close).pack(
            side=tk.RIGHT, padx=8, pady=6
        )
        ttk.Button(header, text="Refresh", command=self.refresh).pack(
            side=tk.RIGHT, padx=4, pady=6
        )

        body = tk.Frame(self, bg=BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: matplotlib chart
        chart_frame = tk.Frame(body, bg=BG)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._fig = Figure(figsize=(7, 4.5), dpi=100, facecolor=BG)
        self._canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right: stats panel
        stats_frame = tk.Frame(body, bg=PANEL_BG, width=300)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        stats_frame.pack_propagate(False)

        tk.Label(
            stats_frame,
            text="Summary",
            bg=PANEL_BG,
            fg=FG,
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 8))

        rows = [
            ("total_images", "Total images"),
            ("annotated_images", "Annotated images"),
            ("unannotated_images", "Unannotated images"),
            ("total_boxes", "Total annotations"),
            ("avg_boxes", "Avg boxes / annotated img"),
            ("imbalance", "Class imbalance ratio"),
            ("refreshed", "Last refreshed"),
        ]
        for key, label in rows:
            row = tk.Frame(stats_frame, bg=PANEL_BG)
            row.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(
                row,
                text=label,
                bg=PANEL_BG,
                fg=MUTED,
                anchor="w",
                font=("TkDefaultFont", 9),
            ).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            self._summary_vars[key] = var
            tk.Label(
                row,
                textvariable=var,
                bg=PANEL_BG,
                fg=FG,
                anchor="e",
                font=("TkDefaultFont", 9, "bold"),
            ).pack(side=tk.RIGHT)

        tk.Label(
            stats_frame,
            text="Tip: keep annotating — click Refresh to update.",
            bg=PANEL_BG,
            fg=MUTED,
            anchor="w",
            wraplength=270,
            justify="left",
            font=("TkDefaultFont", 8, "italic"),
        ).pack(anchor="w", padx=12, pady=(16, 12), side=tk.BOTTOM)

    def refresh(self):
        labels = self.parent_app.labels or []
        stats = compute_dataset_stats(
            self.parent_app.store,
            self.parent_app.image_paths,
            num_classes=max(len(labels), 1),
        )
        self._render_chart(stats, labels)
        self._render_summary(stats)

    def _render_chart(self, stats: DatasetStats, labels: List[str]):
        assert self._fig is not None and self._canvas is not None
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.tick_params(colors=FG)
        ax.yaxis.label.set_color(FG)
        ax.xaxis.label.set_color(FG)
        ax.title.set_color(FG)

        if stats.total_boxes == 0 or not labels:
            ax.text(
                0.5,
                0.5,
                "No annotations yet.\nAnnotate images, then click Refresh.",
                ha="center",
                va="center",
                color=MUTED,
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            x = list(range(len(labels)))
            counts = stats.class_counts[: len(labels)]
            colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in x]
            bars = ax.bar(x, counts, color=colors, edgecolor="#333333")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Annotations", fontsize=10)
            ax.set_title("Annotations per class", fontsize=11)
            ax.grid(axis="y", color="#333333", linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            ymax = max(counts) if counts else 0
            for bar, count in zip(bars, counts):
                if count <= 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.01,
                    str(count),
                    ha="center",
                    va="bottom",
                    color=FG,
                    fontsize=8,
                )

        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _render_summary(self, stats: DatasetStats):
        self._summary_vars["total_images"].set(str(stats.total_images))
        self._summary_vars["annotated_images"].set(str(stats.annotated_images))
        self._summary_vars["unannotated_images"].set(str(stats.unannotated_images))
        self._summary_vars["total_boxes"].set(str(stats.total_boxes))
        self._summary_vars["avg_boxes"].set(f"{stats.avg_boxes_per_annotated:.2f}")
        ratio = stats.imbalance_ratio
        self._summary_vars["imbalance"].set(
            "n/a" if ratio == 1.0 and stats.total_boxes == 0 else f"{ratio:.2f}×"
        )
        self._summary_vars["refreshed"].set(time.strftime("%H:%M:%S"))

    def _on_close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._canvas = None
        self.destroy()
