"""
Hailo-8L detector backends for Raspberry Pi AI Hat.

Provides drop-in replacements for VehicleDetector and PlateDetector
that run inference on the Hailo NPU instead of CPU.

Requirements:
    sudo apt install hailo-all

Model prep:
    Export your YOLOv8 model to HEF format using Hailo's Dataflow Compiler (DFC).
    Use NMS on-chip for best performance:
        hailo optimize model.onnx --hw-arch hailo8l
        hailo compile model.har --hw-arch hailo8l
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import cv2

try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface,
        InferVStreams, ConfigureParams,
    )
except ImportError:
    try:
        from hailort import (
            HEF, VDevice, HailoStreamInterface,
            InferVStreams, ConfigureParams,
        )
    except ImportError:
        HEF = None

from src.vehicleDetector import VehicleDet, _clamp_bbox_xyxy as _clamp_vehicle
from src.plateDetector import PlateDet, _clamp_bbox_xyxy as _clamp_plate


class _HailoDetectorBase:
    """Shared Hailo NPU inference logic for any YOLOv8 HEF model with on-chip NMS."""

    def __init__(
        self,
        model_path: str,
        conf: float,
        classes: Optional[Sequence[int]],
    ) -> None:
        if HEF is None:
            raise ImportError(
                "hailort is not installed. Install with: sudo apt install hailo-all"
            )

        self.conf = float(conf)
        self.classes = set(classes) if classes is not None else None

        self.hef = HEF(model_path)
        self.device = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.device.configure(self.hef, configure_params)[0]

        self.input_vstream_info = self.hef.get_input_vstream_infos()
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        info = self.input_vstream_info[0]
        self.input_h = info.shape[1]
        self.input_w = info.shape[2]

    def _infer(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Run inference, return (raw NMS output for batch item 0, orig_w, orig_h)."""
        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            raise TypeError("image_bgr must be a numpy ndarray.")

        orig_h, orig_w = image_bgr.shape[:2]

        resized = cv2.resize(image_bgr, (self.input_w, self.input_h))
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        input_dict = {self.input_vstream_info[0].name: input_data}
        output_dict = {}
        for info in self.output_vstream_info:
            output_dict[info.name] = np.empty(
                [1] + list(info.shape[1:]), dtype=np.float32
            )

        with InferVStreams(self.network_group, input_dict, output_dict) as pipeline:
            pipeline.send(input_dict)
            result = pipeline.recv()

        out = list(result.values())[0]  # [batch, num_dets, 6]
        return out[0], orig_w, orig_h

    def release(self):
        if hasattr(self, 'device'):
            self.device.release()


class HailoVehicleDetector(_HailoDetectorBase):
    """Drop-in replacement for VehicleDetector using Hailo-8L NPU."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.35,
        classes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(model_path, conf, classes)

    def detect(self, frame_bgr: np.ndarray) -> List[VehicleDet]:
        detections, orig_w, orig_h = self._infer(frame_bgr)
        dets: List[VehicleDet] = []

        for det in detections:
            y1_n, x1_n, y2_n, x2_n, score, cls_id = det
            if score < self.conf:
                continue
            cls_int = int(cls_id)
            if self.classes is not None and cls_int not in self.classes:
                continue

            b = _clamp_vehicle(
                (int(round(x1_n * orig_w)), int(round(y1_n * orig_h)),
                 int(round(x2_n * orig_w)), int(round(y2_n * orig_h))),
                orig_w, orig_h,
            )
            if b is not None:
                dets.append(VehicleDet(bbox=b, conf=float(score), cls=cls_int))

        return dets


class HailoPlateDetector(_HailoDetectorBase):
    """Drop-in replacement for PlateDetector using Hailo-8L NPU."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.30,
        classes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(model_path, conf, classes)

    def detect(self, image_bgr: np.ndarray) -> List[PlateDet]:
        detections, orig_w, orig_h = self._infer(image_bgr)
        dets: List[PlateDet] = []

        for det in detections:
            y1_n, x1_n, y2_n, x2_n, score, cls_id = det
            if score < self.conf:
                continue
            cls_int = int(cls_id)
            if self.classes is not None and cls_int not in self.classes:
                continue

            b = _clamp_plate(
                (int(round(x1_n * orig_w)), int(round(y1_n * orig_h)),
                 int(round(x2_n * orig_w)), int(round(y2_n * orig_h))),
                orig_w, orig_h,
            )
            if b is not None:
                dets.append(PlateDet(bbox=b, conf=float(score), cls=cls_int))

        return dets


def make_hailo_vehicle_detector_from_config(cfg: dict) -> HailoVehicleDetector:
    return HailoVehicleDetector(
        model_path=cfg["model_path"],
        conf=cfg.get("conf", 0.35),
        classes=cfg.get("classes"),
    )


def make_hailo_plate_detector_from_config(cfg: dict) -> HailoPlateDetector:
    return HailoPlateDetector(
        model_path=cfg["model_path"],
        conf=cfg.get("conf", 0.30),
        classes=cfg.get("classes"),
    )
