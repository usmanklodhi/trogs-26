import numpy as np
import pytest

ANNOTATION = """# Angle
0.0
# Boxes
10 10 10 50 40 50 40 10
50 12 50 52 80 52 80 12
90 10 90 50 120 50 120 10
10 100 10 140 40 140 40 100
50 100 50 140 80 140 80 100
not a box
# Transcript
ΑΒ *Γ
Δ*Ε
# Lines
0 30 130 30
"""


@pytest.fixture
def annotation_file(tmp_path):
    path = tmp_path / "IAS_1_Rotation1_300dpi_letters.txt"
    path.write_text(ANNOTATION, encoding="utf-8")
    return path


@pytest.fixture
def image():
    rng = np.random.RandomState(0)
    return (rng.rand(200, 200, 3) * 255).astype(np.uint8)
