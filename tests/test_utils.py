from auto_annotator.utils import load_labels_file


def test_load_labels_plain_names(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("dog\ncat\nbird\n")
    assert load_labels_file(p) == ["dog", "cat", "bird"]


def test_load_labels_strips_numeric_id_prefix(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("0 dog\n1 cat\n")
    assert load_labels_file(p) == ["dog", "cat"]


def test_load_labels_skips_blank_and_whitespace_lines(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("dog\n\ncat\n   \nbird\n")
    assert load_labels_file(p) == ["dog", "cat", "bird"]


def test_load_labels_strips_trailing_whitespace(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("dog  \ncat\t\n")
    assert load_labels_file(p) == ["dog", "cat"]


def test_load_labels_keeps_multiword_name_after_id(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("0 high vis vest\n")
    assert load_labels_file(p) == ["high vis vest"]
