import numpy as np
import pytest
import torch

from src.lprReader import (
    ctc_greedy_decode,
    CHARS,
    DEFAULT_INPUT_SIZE,
)


# ---------------------------------------------------------------------------
# ctc_greedy_decode
# ---------------------------------------------------------------------------

class TestCTCGreedyDecode:

    def test_simple_sequence(self):
        # 3 timesteps, 5 classes: blank=0, '0'=1, '1'=2, '2'=3, '3'=4
        chars = ['-', '0', '1', '2', '3']
        logits = np.array([
            [0, 10, 0, 0, 0],   # '0'
            [0, 0, 10, 0, 0],   # '1'
            [0, 0, 0, 10, 0],   # '2'
        ], dtype=np.float32)
        assert ctc_greedy_decode(logits, chars) == '012'

    def test_collapse_repeats(self):
        chars = ['-', 'A', 'B']
        logits = np.array([
            [0, 10, 0],  # A
            [0, 10, 0],  # A (repeat, collapse)
            [0, 0, 10],  # B
            [0, 0, 10],  # B (repeat, collapse)
        ], dtype=np.float32)
        assert ctc_greedy_decode(logits, chars) == 'AB'

    def test_blank_separator_allows_repeat(self):
        # Use 4 classes to avoid square-matrix auto-transpose ambiguity
        chars = ['-', 'A', 'B', 'C']
        logits = np.array([
            [-10, 10, -10, -10],   # A
            [10, -10, -10, -10],   # blank
            [-10, 10, -10, -10],   # A (not collapsed — blank separated)
        ], dtype=np.float32)
        assert ctc_greedy_decode(logits, chars) == 'AA'

    def test_all_blanks(self):
        chars = ['-', 'A', 'B']
        logits = np.array([
            [10, 0, 0],
            [10, 0, 0],
            [10, 0, 0],
        ], dtype=np.float32)
        assert ctc_greedy_decode(logits, chars) == ''

    def test_transposed_input(self):
        # Input shape (num_classes, timesteps) should be auto-transposed
        chars = ['-', 'X', 'Y']
        logits = np.array([
            [0, 0, 0, 0],     # blank scores
            [10, 0, 10, 0],   # X scores
            [0, 10, 0, 10],   # Y scores
        ], dtype=np.float32)
        # Shape is (3, 4) = (num_classes, timesteps) — should transpose
        assert logits.shape[0] == len(chars)
        result = ctc_greedy_decode(logits, chars)
        assert result == 'XYXY'

    def test_single_timestep(self):
        chars = ['-', 'A']
        logits = np.array([[0, 10]], dtype=np.float32)
        assert ctc_greedy_decode(logits, chars) == 'A'

    def test_single_timestep_blank(self):
        chars = ['-', 'A']
        logits = np.array([[10, 0]], dtype=np.float32)
        assert ctc_greedy_decode(logits, chars) == ''

    def test_real_chars_set(self):
        # Use the actual CHARS from the module
        assert CHARS[0] == '-'  # blank
        assert '0' in CHARS
        assert 'A' in CHARS
        assert 'Z' in CHARS
        assert len(CHARS) == 37  # blank + 10 digits + 26 letters

    def test_mixed_blanks_and_chars(self):
        chars = ['-', 'A', 'B', 'C']
        logits = np.array([
            [10, 0, 0, 0],  # blank
            [0, 10, 0, 0],  # A
            [0, 10, 0, 0],  # A (repeat)
            [10, 0, 0, 0],  # blank
            [0, 0, 10, 0],  # B
            [10, 0, 0, 0],  # blank
            [0, 0, 0, 10],  # C
        ], dtype=np.float32)
        assert ctc_greedy_decode(logits, chars) == 'ABC'


# ---------------------------------------------------------------------------
# USLPRNet model architecture
# ---------------------------------------------------------------------------

class TestUSLPRNet:

    def test_model_imports(self):
        from src.lprReader import USLPRNet, SmallBasicBlock
        assert USLPRNet is not None
        assert SmallBasicBlock is not None

    def test_model_output_shape(self):
        from src.lprReader import USLPRNet
        model = USLPRNet(num_classes=37)
        model.eval()
        # Input: [batch=1, channels=3, height=75, width=300]
        x = torch.randn(1, 3, 75, 300)
        with torch.no_grad():
            out = model(x)
        # Output should be [T, B, C] = [timesteps, 1, 37]
        assert out.ndim == 3
        assert out.shape[1] == 1   # batch
        assert out.shape[2] == 37  # num_classes

    def test_model_output_varies_with_width(self):
        from src.lprReader import USLPRNet
        model = USLPRNet(num_classes=37)
        model.eval()
        with torch.no_grad():
            out1 = model(torch.randn(1, 3, 75, 300))
            out2 = model(torch.randn(1, 3, 75, 150))
        # Different widths should produce different number of timesteps
        assert out1.shape[0] != out2.shape[0]

    def test_model_batch_size(self):
        from src.lprReader import USLPRNet
        model = USLPRNet(num_classes=37)
        model.eval()
        x = torch.randn(4, 3, 75, 300)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == 4  # batch=4

    def test_small_basic_block_skip_connection(self):
        from src.lprReader import SmallBasicBlock
        # When in_ch != out_ch, skip connection should exist
        block = SmallBasicBlock(64, 128)
        assert block.skip is not None
        # When in_ch == out_ch, skip should be None
        block_same = SmallBasicBlock(128, 128)
        assert block_same.skip is None

    def test_small_basic_block_output_shape(self):
        from src.lprReader import SmallBasicBlock
        block = SmallBasicBlock(64, 128)
        x = torch.randn(1, 64, 10, 20)
        out = block(x)
        assert out.shape == (1, 128, 10, 20)


# ---------------------------------------------------------------------------
# PyTorchLPRReader
# ---------------------------------------------------------------------------

class TestPyTorchLPRReader:

    def test_preprocess_output_shape(self):
        from src.lprReader import PyTorchLPRReader, USLPRNet
        # Create a reader with a temp model
        import tempfile, os
        model = USLPRNet(num_classes=37)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            tmp_path = f.name
        try:
            reader = PyTorchLPRReader(model_path=tmp_path)
            img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            tensor = reader._preprocess(img)
            assert tensor.shape == (1, 3, 75, 300)
            assert tensor.dtype == torch.float32
            assert tensor.min() >= 0.0
            assert tensor.max() <= 1.0
        finally:
            os.unlink(tmp_path)

    def test_read_returns_string(self):
        from src.lprReader import PyTorchLPRReader, USLPRNet
        import tempfile, os
        model = USLPRNet(num_classes=37)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            tmp_path = f.name
        try:
            reader = PyTorchLPRReader(model_path=tmp_path)
            img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            result = reader.read(img)
            assert isinstance(result, str)
        finally:
            os.unlink(tmp_path)

    def test_read_empty_image_returns_empty(self):
        from src.lprReader import PyTorchLPRReader, USLPRNet
        import tempfile, os
        model = USLPRNet(num_classes=37)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            tmp_path = f.name
        try:
            reader = PyTorchLPRReader(model_path=tmp_path)
            assert reader.read(None) == ''
            assert reader.read(np.array([])) == ''
        finally:
            os.unlink(tmp_path)

    def test_read_uses_correct_indexing(self):
        """Verify [T, B, C] output is correctly indexed as [:, 0, :]."""
        from src.lprReader import PyTorchLPRReader, USLPRNet
        import tempfile, os
        model = USLPRNet(num_classes=37)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            tmp_path = f.name
        try:
            reader = PyTorchLPRReader(model_path=tmp_path)
            img = np.random.randint(0, 255, (75, 300, 3), dtype=np.uint8)
            preprocessed = reader._preprocess(img)
            with torch.no_grad():
                logits = reader.model(preprocessed)
            # Verify output is [T, B, C]
            assert logits.ndim == 3
            assert logits.shape[1] == 1  # batch
            # The indexing [:, 0, :] should give [T, C]
            logits_np = logits[:, 0, :].cpu().numpy()
            assert logits_np.ndim == 2
            assert logits_np.shape[1] == 37  # num_classes
        finally:
            os.unlink(tmp_path)

    def test_strict_loading_rejects_mismatched_weights(self):
        from src.lprReader import PyTorchLPRReader, USLPRNet
        import tempfile, os
        # Save weights with different num_classes
        model = USLPRNet(num_classes=10)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            tmp_path = f.name
        try:
            with pytest.raises(RuntimeError):
                # Default chars has 37 classes, but weights are for 10
                PyTorchLPRReader(model_path=tmp_path)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# DEFAULT_INPUT_SIZE
# ---------------------------------------------------------------------------

def test_default_input_size():
    assert DEFAULT_INPUT_SIZE == (300, 75)
