"""Tests for main.py helper functions and config loading."""

import pytest
import numpy as np

from main import load_config


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:

    def test_loads_valid_yaml(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("camera:\n  fps: 10\n")
        config = load_config(str(cfg))
        assert config['camera']['fps'] == 10

    def test_expands_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_URL", "rtsp://example.com")
        cfg = tmp_path / "test.yaml"
        cfg.write_text("camera:\n  source: ${TEST_URL}\n")
        config = load_config(str(cfg))
        assert config['camera']['source'] == "rtsp://example.com"

    def test_missing_env_var_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MISSING_VAR_XYZ", raising=False)
        cfg = tmp_path / "test.yaml"
        cfg.write_text("camera:\n  source: ${MISSING_VAR_XYZ}\n")
        with pytest.raises(EnvironmentError, match="Missing environment variables"):
            load_config(str(cfg))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")


# ---------------------------------------------------------------------------
# OCRPlateReader
# ---------------------------------------------------------------------------

class TestOCRPlateReader:

    def test_reads_synthetic_plate(self):
        import cv2
        from src.lprReader import OCRPlateReader

        reader = OCRPlateReader()
        plate = np.ones((80, 240, 3), dtype=np.uint8) * 255
        cv2.putText(plate, "ABC1234", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        result = reader.read(plate)
        assert len(result) >= 4
        assert all(c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in result)

    def test_returns_empty_on_blank(self):
        from src.lprReader import OCRPlateReader

        reader = OCRPlateReader()
        blank = np.zeros((80, 240, 3), dtype=np.uint8)
        assert reader.read(blank) == ''

    def test_returns_empty_on_none(self):
        from src.lprReader import OCRPlateReader

        reader = OCRPlateReader()
        assert reader.read(None) == ''
