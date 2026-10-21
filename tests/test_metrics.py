from trogs.evaluation.contest_metrics import convert, edit_operations, evaluate, get_row_transcript


def test_convert_transliterates_greek():
    assert convert("αβγ ΑΩ") == "ABG AW"


def test_evaluate_and_edit_operations():
    cer, dist, n = evaluate("αβ γ", "ABDE")
    assert (dist, n) == (2, 4) and cer == 0.5
    assert edit_operations("αβγ", "ABDE") == (0, 1, 1)


def test_row_transcript_follows_reading_order(annotation_file):
    # Boxes with the transcript letters ('*' removed), regrouped into rows
    assert get_row_transcript(str(annotation_file)) == "ΑΒΓ\nΔΕ\n"
