"""
License plate reader using fast-plate-ocr.

Takes a cropped plate image and returns the plate text string.

Usage:
    reader = OCRPlateReader()
    text = reader.read(plate_crop_bgr)  # e.g. "ABC1234"
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from fast_plate_ocr import LicensePlateRecognizer


class OCRPlateReader:

    _ALLOWED = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def __init__(self, cooldown: float = 2.0, **_kwargs) -> None:
        self._recognizer = LicensePlateRecognizer(
            hub_ocr_model='global-plates-mobile-vit-v2-model',
            device='cpu',
        )
        self._last_plate = ''
        self._last_read_time = 0.0
        self._cooldown = cooldown

    def read(self, plate_crop: np.ndarray) -> str:
        """Returns plate text, or '' if skipped (cooldown/dedup)."""
        if plate_crop is None or plate_crop.size == 0:
            return ''

        now = time.time()
        if now - self._last_read_time < self._cooldown:
            return ''

        self._last_read_time = now

        if plate_crop.ndim == 3 and plate_crop.shape[2] == 3:
            plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        results = self._recognizer.run(plate_crop)
        if not results:
            return ''

        text = results[0].strip().upper()
        plate = ''.join(c for c in text if c in self._ALLOWED)

        if plate and plate == self._last_plate:
            return ''

        self._last_plate = plate
        return plate

    @property
    def last_plate(self) -> str:
        """Last successfully read plate — useful for display overlay."""
        return self._last_plate
