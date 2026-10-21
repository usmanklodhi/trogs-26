from trogs.inference.reconstruction import chain_rows, reconstruct_transcript, row_ioa, sort_rows


def box(x, y, label, w=20, h=30):
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h, "label": label}


def test_two_lines_in_reading_order():
    letters = [box(60, 102, "E"), box(10, 10, "A"), box(35, 12, "B"), box(10, 100, "D"), box(35, 101, "Δ")]
    assert reconstruct_transcript(letters) == "AB\nDΔE"


def test_empty():
    assert reconstruct_transcript([]) == ""


def test_row_ioa():
    row1 = [box(0, 0, "a", w=10, h=10)]
    row2 = [box(5, 0, "b", w=10, h=10)]
    assert row_ioa(row1, row2) == (0.5, 0.5)


def test_chain_rows_uses_tolerance_radius():
    boxes = [{"cx": 0, "cy": 0, "r": 5}, {"cx": 10, "cy": 4, "r": 5}, {"cx": 20, "cy": 50, "r": 5}]
    rows = sort_rows(chain_rows(boxes))
    assert [[b["id"] for b in row] for row in rows] == [[0, 1], [2]]
