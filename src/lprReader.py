"""
LPRNet license plate reader wrapper.

Takes a cropped plate image and returns the plate text string.
Supports three backends:
  - PyTorch (.pth): for CPU inference with trained weights
  - ONNX (.onnx): uses onnxruntime
  - Hailo (.hef): uses Hailo-8L NPU

LPRNet outputs a sequence of character logits. We apply CTC greedy
decoding (collapse repeats, remove blanks) to get the final text.

Usage:
    reader = PyTorchLPRReader("models/lpr/lprnet.pth")
    text = reader.read(plate_crop_bgr)  # e.g. "7ABC123"
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cv2


# Character set — index 0 is CTC blank, rest matches training
CHARS = ['-'] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Input size matching training config
DEFAULT_INPUT_SIZE = (300, 75)  # (width, height)


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


# ---------------------------------------------------------------------------
# USLPRNet model architecture (must match training code exactly)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    class SmallBasicBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            mid = out_ch // 4

            self.block = nn.Sequential(
                nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid, mid, kernel_size=(3, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid, mid, kernel_size=(1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

            self.skip = None
            if in_ch != out_ch:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            identity = x if self.skip is None else self.skip(x)
            return self.relu(self.block(x) + identity)

    class USLPRNet(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1),

                SmallBasicBlock(64, 128),
                SmallBasicBlock(128, 128),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1),

                SmallBasicBlock(128, 256),
                SmallBasicBlock(256, 256),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1),

                SmallBasicBlock(256, 256),
                nn.Conv2d(256, 256, kernel_size=(4, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, num_classes, kernel_size=1),
            )

        def forward(self, x):
            x = self.backbone(x)       # [B, C, H, W]
            x = x.mean(dim=2)          # [B, C, W]
            x = x.permute(2, 0, 1)     # [T, B, C]
            return x

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# PyTorch backend (.pth)
# ---------------------------------------------------------------------------
class PyTorchLPRReader:
    """
    Reads license plate text using a PyTorch LPRNet .pth weights file.
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        chars: Optional[list] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")

        self.chars = chars or CHARS
        self.input_w, self.input_h = input_size

        self.device = torch.device('cpu')
        self.model = USLPRNet(num_classes=len(self.chars))

        state = torch.load(model_path, map_location=self.device, weights_only=False)
        # Handle both raw state_dict and wrapped checkpoint
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        elif isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def read(self, plate_crop: np.ndarray) -> str:
        if plate_crop is None or plate_crop.size == 0:
            return ''

        preprocessed = self._preprocess(plate_crop)
        with torch.no_grad():
            logits = self.model(preprocessed)

        # USLPRNet returns [T, B, C] — extract batch item 0 → [T, C]
        logits_np = logits[:, 0, :].cpu().numpy()
        return ctc_greedy_decode(logits_np, self.chars)

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(img, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        tensor = torch.from_numpy(transposed).unsqueeze(0).to(self.device)
        return tensor


# ---------------------------------------------------------------------------
# ONNX backend (.onnx)
# ---------------------------------------------------------------------------
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
        if plate_crop is None or plate_crop.size == 0:
            return ''

        preprocessed = self._preprocess(plate_crop)

        outputs = self.session.run(None, {self.input_name: preprocessed})
        logits = outputs[0]

        if logits.ndim == 3:
            logits = logits[0]

        return ctc_greedy_decode(logits, self.chars)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0)


# ---------------------------------------------------------------------------
# Hailo backend (.hef)
# ---------------------------------------------------------------------------
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
