import datetime
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from auto_annotator.models import BoundingBox


def split_image_paths(
    paths: List[Path],
    train_pct: int,
    val_pct: int,
    test_pct: int,
    seed: int = 42,
) -> Dict[str, List[Path]]:
    if train_pct + val_pct + test_pct != 100:
        raise ValueError(
            f"Split ratios must sum to 100, got {train_pct + val_pct + test_pct}"
        )
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = round(n * test_pct / 100)
    n_val = round(n * val_pct / 100)
    n_train = n - n_val - n_test
    result: Dict[str, List[Path]] = {}
    idx = 0
    if train_pct > 0:
        result["train"] = shuffled[idx : idx + n_train]
        idx += n_train
    if val_pct > 0:
        result["val"] = shuffled[idx : idx + n_val]
        idx += n_val
    if test_pct > 0:
        result["test"] = shuffled[idx : idx + n_test]
    return result


class AnnotationStore:
    def __init__(self):
        self._store: Dict[Path, List[BoundingBox]] = {}
        self._dirty: set = set()

    def set_boxes(self, path: Path, boxes: List[BoundingBox]):
        self._store[path] = boxes
        self._dirty.add(path)

    def get_boxes(self, path: Path) -> List[BoundingBox]:
        return self._store.get(path, [])

    def is_annotated(self, path: Path) -> bool:
        return path in self._store

    def is_dirty(self, path: Path) -> bool:
        return path in self._dirty

    def save(self, path: Path):
        boxes = self._store.get(path, [])
        txt_path = path.with_suffix(".txt")
        tmp_path = txt_path.with_suffix(".txt.tmp")
        lines = [f"{b.class_id} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes]
        tmp_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        os.replace(tmp_path, txt_path)
        self._dirty.discard(path)

    def save_all(self) -> int:
        count = 0
        for path in list(self._store.keys()):
            self.save(path)
            count += 1
        return count

    def save_dirty(self) -> int:
        count = 0
        for path in list(self._dirty):
            self.save(path)
            count += 1
        return count

    def load_existing(self, path: Path) -> Optional[List[BoundingBox]]:
        txt_path = path.with_suffix(".txt")
        if not txt_path.exists():
            return None
        boxes = []
        for line in txt_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cid = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append(BoundingBox(cid, cx, cy, w, h, confidence=1.0))
        self._store[path] = boxes
        return boxes

    @staticmethod
    def _build_class_map(
        imported_labels: List[str], current_labels: List[str]
    ) -> Tuple[Dict[int, int], List[str]]:
        lookup = {name.lower(): idx for idx, name in enumerate(current_labels)}
        class_map: Dict[int, int] = {}
        skipped: List[str] = []
        for i, name in enumerate(imported_labels):
            key = name.strip().lower()
            if key in lookup:
                class_map[i] = lookup[key]
            else:
                skipped.append(name.strip())
        return class_map, skipped

    def import_yolo(
        self,
        yolo_dir: Path,
        image_paths: List[Path],
        current_labels: List[str],
    ) -> Tuple[int, int, List[str]]:
        classes_path = yolo_dir / "classes.txt"
        if not classes_path.exists():
            raise FileNotFoundError(
                f"classes.txt not found in {yolo_dir}\n\n"
                "Place a classes.txt file (one class name per line) "
                "alongside your .txt label files."
            )
        imported_labels = [
            ln.strip() for ln in classes_path.read_text().splitlines() if ln.strip()
        ]
        class_map, skipped = self._build_class_map(imported_labels, current_labels)

        stem_to_path = {p.stem: p for p in image_paths}

        labels_dir = yolo_dir / "labels"
        search_dir = labels_dir if labels_dir.is_dir() else yolo_dir

        n_images = 0
        n_boxes = 0
        for txt_path in sorted(search_dir.glob("*.txt")):
            if txt_path.name == "classes.txt":
                continue
            img_path = stem_to_path.get(txt_path.stem)
            if img_path is None:
                continue
            new_boxes: List[BoundingBox] = []
            for line in txt_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                old_cid = int(parts[0])
                if old_cid not in class_map:
                    continue
                cx, cy, w, h = (
                    float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]),
                )
                new_boxes.append(
                    BoundingBox(class_map[old_cid], cx, cy, w, h, confidence=1.0)
                )
            if new_boxes:
                existing = self._store.get(img_path, [])
                existing.extend(new_boxes)
                self._store[img_path] = existing
                self._dirty.add(img_path)
                n_images += 1
                n_boxes += len(new_boxes)
        return n_images, n_boxes, skipped

    def import_coco(
        self,
        coco_dir: Path,
        image_paths: List[Path],
        current_labels: List[str],
    ) -> Tuple[int, int, List[str]]:
        ann_path = coco_dir / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"annotations.json not found in {coco_dir}\n\n"
                "Place a COCO-format annotations.json file in this directory."
            )
        data = json.loads(ann_path.read_text())

        coco_id_to_name = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
        imported_labels = list(coco_id_to_name.values())
        coco_ids = list(coco_id_to_name.keys())
        class_map: Dict[int, int] = {}
        skipped: List[str] = []
        lookup = {name.lower(): idx for idx, name in enumerate(current_labels)}
        for coco_id, name in zip(coco_ids, imported_labels):
            key = name.strip().lower()
            if key in lookup:
                class_map[coco_id] = lookup[key]
            else:
                skipped.append(name.strip())

        img_id_to_info: Dict[int, dict] = {}
        for img in data.get("images", []):
            img_id_to_info[img["id"]] = img

        stem_to_path = {p.stem: p for p in image_paths}
        name_to_path = {p.name: p for p in image_paths}

        anns_by_image: Dict[int, List[dict]] = {}
        for ann in data.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        n_images = 0
        n_boxes = 0
        for img_id, anns in anns_by_image.items():
            info = img_id_to_info.get(img_id)
            if info is None:
                continue
            fname = info["file_name"]
            img_path = name_to_path.get(fname) or stem_to_path.get(Path(fname).stem)
            if img_path is None:
                continue
            iw = info["width"]
            ih = info["height"]
            if iw <= 0 or ih <= 0:
                continue

            new_boxes: List[BoundingBox] = []
            for ann in anns:
                coco_cid = ann["category_id"]
                if coco_cid not in class_map:
                    continue
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw / 2) / iw
                cy = (y + bh / 2) / ih
                nw = bw / iw
                nh = bh / ih
                new_boxes.append(
                    BoundingBox(class_map[coco_cid], cx, cy, nw, nh, confidence=1.0)
                )
            if new_boxes:
                existing = self._store.get(img_path, [])
                existing.extend(new_boxes)
                self._store[img_path] = existing
                self._dirty.add(img_path)
                n_images += 1
                n_boxes += len(new_boxes)
        return n_images, n_boxes, skipped

    def _export_coco_subset(
        self, image_paths: List[Path], labels: List[str], images_out: Path,
    ) -> Tuple[List[dict], List[dict]]:
        images_out.mkdir(parents=True, exist_ok=True)
        coco_images: List[dict] = []
        coco_anns: List[dict] = []
        ann_id = 0
        img_id = 0
        for path in image_paths:
            if not self.is_annotated(path):
                continue
            try:
                with Image.open(path) as pil:
                    iw, ih = pil.size
            except Exception:
                continue
            dest = images_out / path.name
            if dest.resolve() != path.resolve():
                shutil.copy2(str(path), str(dest))
            coco_images.append({
                "id": img_id,
                "file_name": path.name,
                "width": iw,
                "height": ih,
                "license": 0,
            })
            for box in self.get_boxes(path):
                abs_w = box.w * iw
                abs_h = box.h * ih
                abs_x = box.cx * iw - abs_w / 2
                abs_y = box.cy * ih - abs_h / 2
                coco_anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": box.class_id,
                    "bbox": [abs_x, abs_y, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "segmentation": [],
                    "iscrowd": 0,
                })
                ann_id += 1
            img_id += 1
        return coco_images, coco_anns

    @staticmethod
    def _build_coco_payload(
        categories: List[dict], coco_images: List[dict], coco_anns: List[dict],
    ) -> dict:
        return {
            "info": {
                "description": "Annotations",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "date_created": datetime.datetime.now().isoformat(),
            },
            "licenses": [{"id": 0, "name": "Unknown", "url": ""}],
            "images": coco_images,
            "annotations": coco_anns,
            "categories": categories,
        }

    def export_coco(
        self,
        image_paths: List[Path],
        labels: List[str],
        output_dir: Path,
        splits: Optional[Dict[str, List[Path]]] = None,
    ) -> Tuple[int, int]:
        categories = [
            {"id": i, "name": lbl, "supercategory": "object"}
            for i, lbl in enumerate(labels)
        ]
        if splits is None:
            images_out = output_dir / "images"
            coco_images, coco_anns = self._export_coco_subset(
                image_paths, labels, images_out,
            )
            payload = self._build_coco_payload(categories, coco_images, coco_anns)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "annotations.json").write_text(
                json.dumps(payload, indent=2)
            )
            return len(coco_images), len(coco_anns)

        total_images = 0
        total_anns = 0
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_paths in splits.items():
            split_dir = output_dir / split_name
            images_out = split_dir / "images"
            coco_images, coco_anns = self._export_coco_subset(
                split_paths, labels, images_out,
            )
            payload = self._build_coco_payload(categories, coco_images, coco_anns)
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "annotations.json").write_text(
                json.dumps(payload, indent=2)
            )
            total_images += len(coco_images)
            total_anns += len(coco_anns)
        return total_images, total_anns

    def _export_yolo_subset(
        self, image_paths: List[Path], images_out: Path, labels_out: Path,
    ) -> int:
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        count = 0
        for path in image_paths:
            if not self.is_annotated(path):
                continue
            boxes = self.get_boxes(path)
            if not boxes:
                continue
            dest = images_out / path.name
            if dest.resolve() != path.resolve():
                shutil.copy2(str(path), str(dest))
            lines = [
                f"{b.class_id} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}"
                for b in boxes
            ]
            (labels_out / (path.stem + ".txt")).write_text("\n".join(lines) + "\n")
            count += 1
        return count

    def export_yolo(
        self,
        image_paths: List[Path],
        labels: List[str],
        output_dir: Path,
        splits: Optional[Dict[str, List[Path]]] = None,
    ) -> int:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "classes.txt").write_text("\n".join(labels) + "\n")
        names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(labels))

        if splits is None:
            images_out = output_dir / "images"
            labels_out = output_dir / "labels"
            (output_dir / "data.yaml").write_text(
                f"path: {output_dir.resolve()}\n"
                f"train: images\n"
                f"val: images\n"
                f"\n"
                f"nc: {len(labels)}\n"
                f"names:\n"
                f"{names_yaml}\n"
            )
            return self._export_yolo_subset(image_paths, images_out, labels_out)

        yaml_lines = [f"path: {output_dir.resolve()}"]
        for split_name in ("train", "val", "test"):
            if split_name in splits:
                yaml_lines.append(f"{split_name}: {split_name}/images")
        yaml_lines.append("")
        yaml_lines.append(f"nc: {len(labels)}")
        yaml_lines.append("names:")
        yaml_lines.append(names_yaml)
        (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

        total = 0
        for split_name, split_paths in splits.items():
            split_dir = output_dir / split_name
            total += self._export_yolo_subset(
                split_paths, split_dir / "images", split_dir / "labels",
            )
        return total
