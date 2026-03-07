import numpy as np
import pytest
from src.crop import crop_bbox


@pytest.fixture
def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_basic_crop(frame):
    crop = crop_bbox(frame, (10, 20, 100, 200))
    assert crop is not None
    assert crop.shape == (180, 90, 3)


def test_crop_full_frame(frame):
    # bbox coords are clamped to w-1, h-1, so (640,480) -> (639,479)
    crop = crop_bbox(frame, (0, 0, 640, 480))
    assert crop is not None
    assert crop.shape == (479, 639, 3)


def test_crop_clamps_to_boundaries(frame):
    crop = crop_bbox(frame, (-10, -10, 50, 50))
    assert crop is not None
    assert crop.shape == (50, 50, 3)


def test_crop_clamps_large_coords(frame):
    crop = crop_bbox(frame, (600, 400, 9999, 9999))
    assert crop is not None
    # x clamped to 600..639, y to 400..479
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0


def test_crop_zero_width_returns_none(frame):
    result = crop_bbox(frame, (50, 50, 50, 100))
    assert result is None


def test_crop_zero_height_returns_none(frame):
    result = crop_bbox(frame, (50, 50, 100, 50))
    assert result is None


def test_crop_inverted_bbox_returns_none(frame):
    result = crop_bbox(frame, (100, 100, 50, 50))
    assert result is None


def test_crop_completely_outside_returns_none(frame):
    result = crop_bbox(frame, (700, 500, 800, 600))
    assert result is None


def test_crop_preserves_pixel_values():
    frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    crop = crop_bbox(frame, (10, 20, 30, 40))
    expected = frame[20:40, 10:30]
    np.testing.assert_array_equal(crop, expected)


def test_crop_none_image_raises():
    with pytest.raises(TypeError):
        crop_bbox(None, (0, 0, 10, 10))


def test_crop_non_array_raises():
    with pytest.raises(TypeError):
        crop_bbox("not an image", (0, 0, 10, 10))


def test_crop_single_pixel():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    crop = crop_bbox(frame, (5, 5, 6, 6))
    assert crop is not None
    assert crop.shape == (1, 1, 3)
