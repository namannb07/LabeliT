import json
from pathlib import Path

import pytest
from PIL import Image

from auto_annotator.models import BoundingBox
from auto_annotator.store import AnnotationStore, split_image_paths


def make_image(path: Path, size=(640, 480)):
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def test_set_and_get_boxes(tmp_path):
    store = AnnotationStore()
    img = tmp_path / "a.jpg"
    boxes = [BoundingBox(0, 0.5, 0.5, 0.2, 0.2, 0.9)]
    store.set_boxes(img, boxes)
    assert store.get_boxes(img) == boxes
    assert store.is_annotated(img)
    assert store.is_dirty(img)


def test_get_boxes_for_unknown_path_returns_empty_list():
    assert AnnotationStore().get_boxes(Path("/nonexistent.jpg")) == []


def test_save_writes_yolo_txt_and_clears_dirty(tmp_path):
    store = AnnotationStore()
    img = tmp_path / "img.jpg"
    store.set_boxes(
        img,
        [
            BoundingBox(1, 0.5, 0.5, 0.2, 0.3, 0.9),
            BoundingBox(0, 0.25, 0.75, 0.1, 0.1, 0.8),
        ],
    )
    store.save(img)
    lines = img.with_suffix(".txt").read_text().splitlines()
    assert lines == [
        "1 0.500000 0.500000 0.200000 0.300000",
        "0 0.250000 0.750000 0.100000 0.100000",
    ]
    assert not store.is_dirty(img)


def test_load_existing_parses_yolo_format(tmp_path):
    img = tmp_path / "img.jpg"
    img.with_suffix(".txt").write_text(
        "2 0.5 0.5 0.1 0.1\n0 0.25 0.25 0.05 0.05\n"
    )
    store = AnnotationStore()
    boxes = store.load_existing(img)
    assert boxes is not None
    assert len(boxes) == 2
    assert boxes[0].class_id == 2
    assert boxes[1].class_id == 0


def test_load_existing_returns_none_when_no_txt(tmp_path):
    assert AnnotationStore().load_existing(tmp_path / "absent.jpg") is None


def test_export_yolo_writes_classes_labels_and_data_yaml(tmp_path):
    img = tmp_path / "scene.jpg"
    make_image(img)
    store = AnnotationStore()
    store.set_boxes(img, [BoundingBox(0, 0.5, 0.5, 0.4, 0.4, 0.9)])

    out = tmp_path / "out_yolo"
    count = store.export_yolo([img], ["dog"], out)

    assert count == 1
    assert (out / "classes.txt").read_text().strip() == "dog"

    label_lines = (out / "labels" / "scene.txt").read_text().strip().splitlines()
    assert label_lines == ["0 0.500000 0.500000 0.400000 0.400000"]

    assert (out / "images" / "scene.jpg").exists()

    yaml_text = (out / "data.yaml").read_text()
    assert "nc: 1" in yaml_text
    assert "0: dog" in yaml_text


def test_export_coco_writes_correct_bbox_and_image_metadata(tmp_path):
    img = tmp_path / "scene.jpg"
    make_image(img, size=(640, 480))
    store = AnnotationStore()
    store.set_boxes(img, [BoundingBox(0, 0.5, 0.5, 0.4, 0.4, 0.9)])

    out = tmp_path / "out_coco"
    n_images, n_anns = store.export_coco([img], ["cat"], out)
    assert n_images == 1
    assert n_anns == 1

    payload = json.loads((out / "annotations.json").read_text())
    assert payload["categories"] == [
        {"id": 0, "name": "cat", "supercategory": "object"}
    ]
    assert len(payload["images"]) == 1
    assert payload["images"][0]["width"] == 640
    assert payload["images"][0]["height"] == 480

    ann = payload["annotations"][0]
    assert ann["category_id"] == 0
    assert ann["bbox"][2] == pytest.approx(640 * 0.4)
    assert ann["bbox"][3] == pytest.approx(480 * 0.4)


def test_export_yolo_skips_images_without_annotations(tmp_path):
    annotated = tmp_path / "a.jpg"
    unannotated = tmp_path / "b.jpg"
    make_image(annotated)
    make_image(unannotated)

    store = AnnotationStore()
    store.set_boxes(annotated, [BoundingBox(0, 0.5, 0.5, 0.1, 0.1, 0.9)])

    out = tmp_path / "out"
    count = store.export_yolo([annotated, unannotated], ["x"], out)
    assert count == 1
    assert not (out / "images" / "b.jpg").exists()


def test_import_yolo_remaps_class_ids_via_classes_txt(tmp_path):
    yolo_dir = tmp_path / "in_yolo"
    yolo_dir.mkdir()
    (yolo_dir / "classes.txt").write_text("cat\ndog\n")
    img = tmp_path / "scene.jpg"
    make_image(img)
    (yolo_dir / "scene.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    store = AnnotationStore()
    n_imgs, n_boxes, skipped = store.import_yolo(yolo_dir, [img], ["dog", "cat"])

    assert n_imgs == 1
    assert n_boxes == 1
    assert skipped == []
    boxes = store.get_boxes(img)
    assert boxes[0].class_id == 0


def test_import_yolo_reports_unknown_classes_in_skipped_list(tmp_path):
    yolo_dir = tmp_path / "in_yolo"
    yolo_dir.mkdir()
    (yolo_dir / "classes.txt").write_text("ferret\n")
    img = tmp_path / "scene.jpg"
    make_image(img)
    (yolo_dir / "scene.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    store = AnnotationStore()
    n_imgs, _, skipped = store.import_yolo(yolo_dir, [img], ["dog"])
    assert skipped == ["ferret"]
    assert n_imgs == 0


def test_import_yolo_raises_when_classes_txt_missing(tmp_path):
    yolo_dir = tmp_path / "in_yolo"
    yolo_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        AnnotationStore().import_yolo(yolo_dir, [], [])


# --- split_image_paths tests ---


def test_split_basic_counts():
    paths = [Path(f"{i}.jpg") for i in range(10)]
    result = split_image_paths(paths, 70, 20, 10)
    assert len(result["train"]) == 7
    assert len(result["val"]) == 2
    assert len(result["test"]) == 1
    all_split = result["train"] + result["val"] + result["test"]
    assert sorted(all_split) == sorted(paths)


def test_split_deterministic_with_same_seed():
    paths = [Path(f"{i}.jpg") for i in range(20)]
    a = split_image_paths(paths, 70, 20, 10, seed=123)
    b = split_image_paths(paths, 70, 20, 10, seed=123)
    assert a == b


def test_split_different_seeds_differ():
    paths = [Path(f"{i}.jpg") for i in range(20)]
    a = split_image_paths(paths, 70, 20, 10, seed=1)
    b = split_image_paths(paths, 70, 20, 10, seed=2)
    assert a["train"] != b["train"]


def test_split_zero_test():
    paths = [Path(f"{i}.jpg") for i in range(10)]
    result = split_image_paths(paths, 80, 20, 0)
    assert "test" not in result
    assert len(result["train"]) == 8
    assert len(result["val"]) == 2


def test_split_all_train():
    paths = [Path(f"{i}.jpg") for i in range(10)]
    result = split_image_paths(paths, 100, 0, 0)
    assert "val" not in result
    assert "test" not in result
    assert len(result["train"]) == 10


def test_split_rounding():
    paths = [Path(f"{i}.jpg") for i in range(7)]
    result = split_image_paths(paths, 70, 15, 15)
    total = len(result["train"]) + len(result["val"]) + len(result["test"])
    assert total == 7


def test_split_invalid_ratios():
    paths = [Path(f"{i}.jpg") for i in range(5)]
    with pytest.raises(ValueError, match="must sum to 100"):
        split_image_paths(paths, 50, 30, 10)


def test_split_single_image():
    paths = [Path("only.jpg")]
    result = split_image_paths(paths, 70, 20, 10)
    total = sum(len(v) for v in result.values())
    assert total == 1


# --- split export integration tests ---


def test_export_yolo_with_splits(tmp_path):
    store = AnnotationStore()
    imgs = []
    for i in range(4):
        p = tmp_path / f"img{i}.jpg"
        make_image(p)
        store.set_boxes(p, [BoundingBox(0, 0.5, 0.5, 0.2, 0.2, 0.9)])
        imgs.append(p)

    splits = split_image_paths(imgs, 50, 25, 25, seed=42)
    out = tmp_path / "out_yolo_split"
    count = store.export_yolo(imgs, ["dog"], out, splits=splits)

    assert count == 4
    assert (out / "classes.txt").exists()
    yaml_text = (out / "data.yaml").read_text()
    assert "train: train/images" in yaml_text
    assert "val: val/images" in yaml_text

    for split_name, split_paths in splits.items():
        img_dir = out / split_name / "images"
        lbl_dir = out / split_name / "labels"
        assert img_dir.is_dir()
        assert lbl_dir.is_dir()
        assert len(list(img_dir.iterdir())) == len(split_paths)
        assert len(list(lbl_dir.iterdir())) == len(split_paths)


def test_export_coco_with_splits(tmp_path):
    store = AnnotationStore()
    imgs = []
    for i in range(4):
        p = tmp_path / f"img{i}.jpg"
        make_image(p)
        store.set_boxes(p, [BoundingBox(0, 0.5, 0.5, 0.2, 0.2, 0.9)])
        imgs.append(p)

    splits = split_image_paths(imgs, 50, 25, 25, seed=42)
    out = tmp_path / "out_coco_split"
    n_imgs, n_anns = store.export_coco(imgs, ["cat"], out, splits=splits)

    assert n_imgs == 4
    assert n_anns == 4

    for split_name, split_paths in splits.items():
        ann_file = out / split_name / "annotations.json"
        assert ann_file.exists()
        payload = json.loads(ann_file.read_text())
        assert len(payload["images"]) == len(split_paths)
        assert len(payload["annotations"]) == len(split_paths)


def test_export_yolo_without_splits_unchanged(tmp_path):
    img = tmp_path / "a.jpg"
    make_image(img)
    store = AnnotationStore()
    store.set_boxes(img, [BoundingBox(0, 0.5, 0.5, 0.2, 0.2, 0.9)])

    out = tmp_path / "flat_yolo"
    count = store.export_yolo([img], ["x"], out, splits=None)
    assert count == 1
    assert (out / "images" / "a.jpg").exists()
    assert (out / "labels" / "a.txt").exists()
    assert not (out / "train").exists()
