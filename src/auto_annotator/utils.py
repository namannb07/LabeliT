from pathlib import Path
from typing import List


def load_labels_file(labels_path: Path) -> List[str]:
    labels = []
    for line in labels_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        labels.append(parts[-1] if len(parts) == 2 else parts[0])
    return labels
