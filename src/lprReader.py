"""
License plate reader using Tesseract OCR.

Takes a cropped plate image and returns the plate text string.
Lightweight — no PyTorch/ONNX dependency, fast on Pi.

Usage:
    reader = OCRPlateReader()
    text = reader.read(plate_crop_bgr)  # e.g. "ABC1234"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cv2


class OCRPlateReader:
    """
    Reads license plate text using Tesseract OCR.
    Requires: sudo apt install tesseract-ocr (Pi) or brew install tesseract (macOS)
    """

    _ALLOWED = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def __init__(self, **_kwargs) -> None:
        try:
            import pytesseract
            self._pytesseract = pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract is not installed. Install with: pip install pytesseract"
            )
        try:
            self._pytesseract.get_tesseract_version()
        except Exception:
            raise RuntimeError(
                "tesseract binary not found. Install with: "
                "sudo apt install tesseract-ocr (Pi) or brew install tesseract (macOS)"
            )

    def read(self, plate_crop: np.ndarray) -> str:
        if plate_crop is None or plate_crop.size == 0:
            return ''

        processed = self._preprocess(plate_crop)
        text = self._pytesseract.image_to_string(
            processed,
            config='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        )
        return ''.join(c for c in text.strip().upper() if c in self._ALLOWED)

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return gray
