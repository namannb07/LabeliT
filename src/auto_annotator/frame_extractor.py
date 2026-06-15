from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from auto_annotator.config import FRAME_EXTRACT_JPG_QUALITY


@dataclass
class VideoInfo:
    path: Path
    total_frames: int
    fps: float
    width: int
    height: int
    duration_sec: float

    @property
    def resolution(self) -> str:
        return f"{self.width}×{self.height}"


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {path.name}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if total <= 0:
        raise RuntimeError(
            f"Video reports zero frames: {path.name}. "
            "File may be corrupted or use an unsupported codec."
        )

    duration = total / fps if fps > 0 else 0.0
    return VideoInfo(
        path=path,
        total_frames=total,
        fps=fps,
        width=width,
        height=height,
        duration_sec=duration,
    )


def compute_frame_indices(
    total_frames: int, skip: int, cap: Optional[int]
) -> List[int]:
    if total_frames <= 0:
        return []

    stride = max(1, skip + 1)
    pattern = list(range(0, total_frames, stride))

    if cap is None or cap <= 0 or len(pattern) <= cap:
        return pattern

    picks = np.linspace(0, len(pattern) - 1, cap).round().astype(int)
    seen = set()
    out: List[int] = []
    for p in picks:
        idx = pattern[int(p)]
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def extract_frames(
    video_path: Path,
    output_dir: Path,
    indices: List[int],
    on_progress: Callable[[int, int], None],
    on_status: Callable[[str], None],
    should_cancel: Callable[[], bool],
) -> int:
    if not indices:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path.name}")

    wanted = set(indices)
    last_wanted = indices[-1]
    total = len(indices)
    stem = video_path.stem
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_EXTRACT_JPG_QUALITY]

    written = 0
    frame_idx = 0
    on_status("Extracting frames…")

    try:
        while frame_idx <= last_wanted:
            if should_cancel():
                return written

            ok = cap.grab()
            if not ok:
                break

            if frame_idx in wanted:
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    out_path = output_dir / f"{stem}_frame_{frame_idx:06d}.jpg"
                    if cv2.imwrite(str(out_path), frame, encode_params):
                        written += 1
                        on_progress(written, total)

            frame_idx += 1
    finally:
        cap.release()

    return written
