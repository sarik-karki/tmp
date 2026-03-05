# src/plate_detector.py
"""
A yolov8 plate detector wrapper, designed to be used on car crops (ROIs) for speed/accuracy.

Designed to match the VehicleDetector wrapper style:
- Hide Ultralytics Results objects
- Return standardized detections: bbox (xyxy pixel coords), conf, cls
- Intended to run on a CAR CROP (ROI) for speed/accuracy:
      plate_dets = plate_detector.detect(car_crop_bgr)

later we need to implement a hailo backend doing the samething
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None 


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords


@dataclass(frozen=True)
class PlateDet:
    bbox: BBox
    conf: float
    cls: int  # typically a single class (plate), but kept for consistency


def _clamp_bbox_xyxy(b: BBox, w: int, h: int) -> Optional[BBox]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


class PlateDetector:
    """
    Wrapper around an Ultralytics YOLO bbox model trained for license plates.

    Returns:
        List[PlateDet] in pixel coords relative to the *input image*.
        If you pass a car crop, bboxes are relative to that crop (ROI coords).
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.30,
        iou: float = 0.5,
        classes: Optional[Sequence[int]] = None,
        imgsz: Union[int, Tuple[int, int]] = 416,
        device: Optional[Union[int, str]] = None,
        half: bool = False,
        max_det: int = 50,
        agnostic_nms: bool = False,
        verbose: bool = False,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "Ultralytics is not installed. Install with: pip install ultralytics"
            )

        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.classes = list(classes) if classes is not None else None
        self.imgsz = imgsz
        self.device = device
        self.half = bool(half)
        self.max_det = int(max_det)
        self.agnostic_nms = bool(agnostic_nms)
        self.verbose = bool(verbose)

    def detect(self, image_bgr: np.ndarray) -> List[PlateDet]:
        """
        Run plate detection on a single image (typically a car ROI crop).

        Args:
            image_bgr: np.ndarray (H, W, 3) uint8

        Returns:
            List[PlateDet] with bbox coords relative to the given image.
        """
        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            raise TypeError("image_bgr must be a numpy ndarray.")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must have shape (H, W, 3).")

        h, w = image_bgr.shape[:2]

        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            verbose=self.verbose,
        )

        if not results:
            return []

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        xyxy = r0.boxes.xyxy.detach().cpu().numpy()
        conf = r0.boxes.conf.detach().cpu().numpy()
        cls = r0.boxes.cls.detach().cpu().numpy()

        dets: List[PlateDet] = []
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            b = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            b = _clamp_bbox_xyxy(b, w=w, h=h)
            if b is None:
                continue
            dets.append(PlateDet(bbox=b, conf=float(conf[i]), cls=int(cls[i])))

        return dets


def make_plate_detector_from_config(cfg: dict) -> PlateDetector:
    """
    Example cfg:
        {
          "model_path": "models/plates.pt",
          "conf": 0.30,
          "iou": 0.5,
          "imgsz": 416,
          "device": "cpu",
          "classes": [0]   # if your plate model has only one class, optional
        }
    """
    return PlateDetector(
        model_path=cfg["model_path"],
        conf=cfg.get("conf", 0.30),
        iou=cfg.get("iou", 0.5),
        classes=cfg.get("classes"),
        imgsz=cfg.get("imgsz", 416),
        device=cfg.get("device"),
        half=cfg.get("half", False),
        max_det=cfg.get("max_det", 50),
        agnostic_nms=cfg.get("agnostic_nms", False),
        verbose=cfg.get("verbose", False),
    )