"""Tests for OCRPlateReader in src/lprReader.py."""

import numpy as np
import pytest
import cv2

from src.lprReader import OCRPlateReader


class TestOCRPlateReader:

    @pytest.fixture(scope="class")
    def reader(self):
        return OCRPlateReader()

    def test_reads_alphanumeric_only(self, reader):
        plate = np.ones((80, 240, 3), dtype=np.uint8) * 255
        cv2.putText(plate, "ABC1234", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        result = reader.read(plate)
        assert all(c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in result)

    def test_returns_nonempty_for_clear_text(self, reader):
        plate = np.ones((80, 240, 3), dtype=np.uint8) * 255
        cv2.putText(plate, "XYZ789", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        assert len(reader.read(plate)) >= 3

    def test_returns_empty_on_blank(self, reader):
        blank = np.zeros((80, 240, 3), dtype=np.uint8)
        assert reader.read(blank) == ''

    def test_returns_empty_on_none(self, reader):
        assert reader.read(None) == ''

    def test_returns_empty_on_empty_array(self, reader):
        assert reader.read(np.array([])) == ''

    def test_preprocess_output(self):
        img = np.random.randint(0, 255, (80, 240, 3), dtype=np.uint8)
        result = OCRPlateReader._preprocess(img)
        assert result.ndim == 2  # grayscale
        assert result.shape[0] == 80 * 2  # 2x upscale
        assert result.shape[1] == 240 * 2
