"""Tests for main.py helper functions and config loading."""

import os
import pytest
import numpy as np

from main import load_config, read_plate_from_api


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
# read_plate_from_api
# ---------------------------------------------------------------------------

class TestReadPlateFromAPI:

    def test_returns_empty_on_bad_url(self):
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        result = read_plate_from_api("http://127.0.0.1:1", img)
        assert result == ''

    def test_returns_empty_on_empty_url(self):
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        result = read_plate_from_api("", img)
        assert result == ''
