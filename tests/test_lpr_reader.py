"""Tests for OCRPlateReader in src/lprReader.py."""

import numpy as np
import pytest
import cv2

from src.lprReader import OCRPlateReader


class TestOCRPlateReader:

    @pytest.fixture
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

    def test_returns_empty_on_none(self, reader):
        assert reader.read(None) == ''

    def test_returns_empty_on_empty_array(self, reader):
        assert reader.read(np.array([])) == ''

    def test_dedup_same_plate(self, reader):
        plate = np.ones((80, 240, 3), dtype=np.uint8) * 255
        cv2.putText(plate, "ABC1234", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        first = reader.read(plate)
        # Reset cooldown so dedup check runs
        reader._last_read_time = 0.0
        second = reader.read(plate)
        if first:
            assert second == ''  # dedup blocks same plate

    def test_cooldown_skips_read(self, reader):
        plate = np.ones((80, 240, 3), dtype=np.uint8) * 255
        cv2.putText(plate, "TEST123", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        reader.read(plate)
        # Immediate second read should be skipped by cooldown
        result = reader.read(plate)
        assert result == ''
