import numpy as np

from trogs.data.geometry import line_rect, order_points, warp_box, warp_box_square, warp_line


def test_order_points():
    pts = np.array([[10, 50], [40, 10], [10, 10], [40, 50]], dtype=np.float32)
    np.testing.assert_array_equal(order_points(pts), [[10, 10], [40, 10], [40, 50], [10, 50]])


def test_warp_box_sizes(image):
    box = np.array([[20, 30], [20, 90], [60, 90], [60, 30]], dtype=np.float32)
    assert warp_box(image, box, 128).shape == (128, 128, 3)
    assert warp_box_square(image, box, 64).shape == (64, 64, 3)


def test_warp_box_crops_axis_aligned_region(image):
    box = np.array([[20, 30], [20, 90], [60, 90], [60, 30]], dtype=np.float32)
    # A 40x60 box grows to a 60x60 square centred on (40, 60); size 61 makes the warp 1:1.
    square = warp_box_square(image, box, 61)
    np.testing.assert_array_equal(square[30, 30], image[60, 40])


def test_line_rect_is_landscape_and_warp_keeps_height(image):
    points = np.array([[10, 10], [10, 40], [150, 40], [150, 10]], dtype=np.float32)
    center, w, h, _ = line_rect(points)
    assert w >= h
    crop = warp_line(image, center, w + h, h, _, target_h=32)
    assert crop.shape[0] == 32
    assert crop.shape[1] == int((w + h) * 32 / h)
