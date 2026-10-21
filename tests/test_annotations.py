import numpy as np

from trogs.data.annotations import parse_boxes, parse_letters, parse_text_lines


def test_parse_boxes_skips_malformed_lines(annotation_file):
    boxes = parse_boxes(annotation_file)
    assert len(boxes) == 5
    assert boxes[0].shape == (4, 2)
    np.testing.assert_array_equal(boxes[1][0], [50, 12])


def test_parse_letters_drops_placeholders_and_spaces(annotation_file):
    boxes, labels = parse_letters(annotation_file)
    assert labels == ["Α", "Β", "Γ", "Δ", "Ε"]
    assert len(boxes) == len(labels)
    assert boxes[0].dtype == np.float32


def test_parse_letters_truncates_to_shorter_list(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("# Boxes\n0 0 0 1 1 1 1 0\n# Transcript\nΑΒΓ\n", encoding="utf-8")
    boxes, labels = parse_letters(path)
    assert len(boxes) == len(labels) == 1


def test_parse_text_lines_groups_boxes_by_row(annotation_file):
    lines = parse_text_lines(annotation_file)
    assert [line.text for line in lines] == ["ΑΒ*Γ", "Δ*Ε"]
    # '*' consumes no box: 3 boxes in the first row, 2 in the second
    assert [line.points.shape for line in lines] == [(12, 2), (8, 2)]


def test_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "missing.txt"
    assert parse_boxes(missing) == []
    assert parse_letters(missing) == ([], [])
    assert parse_text_lines(missing) == []
