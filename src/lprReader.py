"""
LPRNet license plate reader wrapper.

Takes a cropped plate image and returns the plate text string.
Supports two backends:
  - ONNX (for dev/CPU): uses onnxruntime
  - Hailo (for Pi with AI Hat): uses hailort HEF model

LPRNet outputs a sequence of character logits. We apply CTC greedy
decoding (collapse repeats, remove blanks) to get the final text.

Usage:
    reader = LPRReader("models/lpr/lprnet.onnx")
    text = reader.read(plate_crop_bgr)  # e.g. "7ABC123"
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cv2


# Standard LPRNet character set — update this to match your training
CHARS = [
    '-',  # 0 = CTC blank
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
]

# Standard LPRNet input size
DEFAULT_INPUT_SIZE = (94, 24)  # (width, height)


def ctc_greedy_decode(logits: np.ndarray, chars: list, blank_idx: int = 0) -> str:
    """
    Greedy CTC decode: pick best class per timestep, collapse repeats, remove blanks.

    Args:
        logits: shape (timesteps, num_classes) or (num_classes, timesteps)
        chars: character list indexed by class ID
        blank_idx: index of the CTC blank token
    """
    # Ensure shape is (timesteps, num_classes)
    if logits.ndim == 2 and logits.shape[0] == len(chars):
        logits = logits.T

    best = np.argmax(logits, axis=1)

    # Collapse repeats and remove blanks
    result = []
    prev = -1
    for idx in best:
        if idx != prev and idx != blank_idx:
            if idx < len(chars):
                result.append(chars[idx])
        prev = idx

    return ''.join(result)


class LPRReader:
    """
    Reads license plate text from a cropped plate image.
    Uses ONNX runtime for inference (CPU/dev mode).
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        chars: Optional[list] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is not installed. Install with: pip install onnxruntime"
            )

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_w, self.input_h = input_size
        self.chars = chars or CHARS

        # Read input shape from model if available
        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_h = shape[2]
            self.input_w = shape[3]

    def read(self, plate_crop: np.ndarray) -> str:
        """
        Read plate text from a BGR crop.

        Args:
            plate_crop: np.ndarray (H, W, 3) uint8

        Returns:
            Plate text string, e.g. "7ABC123". Empty string if unreadable.
        """
        if plate_crop is None or plate_crop.size == 0:
            return ''

        preprocessed = self._preprocess(plate_crop)

        outputs = self.session.run(None, {self.input_name: preprocessed})
        logits = outputs[0]

        # Remove batch dim if present
        if logits.ndim == 3:
            logits = logits[0]

        return ctc_greedy_decode(logits, self.chars)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize, normalize, transpose to NCHW float32."""
        resized = cv2.resize(img, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        # HWC -> CHW -> NCHW
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0)


class HailoLPRReader:
    """
    Reads license plate text using Hailo-8L NPU.
    Expects a compiled LPRNet HEF model.
    """

    def __init__(
        self,
        model_path: str,
        chars: Optional[list] = None,
    ) -> None:
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
                raise ImportError(
                    "hailort is not installed. Install with: sudo apt install hailo-all"
                )

        self.chars = chars or CHARS

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

        self._InferVStreams = InferVStreams

    def read(self, plate_crop: np.ndarray) -> str:
        if plate_crop is None or plate_crop.size == 0:
            return ''

        preprocessed = self._preprocess(plate_crop)

        input_dict = {self.input_vstream_info[0].name: preprocessed}
        output_dict = {}
        for info in self.output_vstream_info:
            output_dict[info.name] = np.empty(
                [1] + list(info.shape[1:]), dtype=np.float32
            )

        with self._InferVStreams(self.network_group, input_dict, output_dict) as pipeline:
            pipeline.send(input_dict)
            result = pipeline.recv()

        logits = list(result.values())[0]
        if logits.ndim == 3:
            logits = logits[0]

        return ctc_greedy_decode(logits, self.chars)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0)

    def release(self):
        if hasattr(self, 'device'):
            self.device.release()
