import pytest
import numpy as np
import cv2

from openscan_firmware.utils.quality_gate import (
    QualityGate,
    QualityResult,
    DEFAULT_MIN_LAPLACIAN,
    DEFAULT_DOWNSAMPLE_FACTOR,
)

# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def make_checkerboard(height=868, width=1152, tile=32) -> np.ndarray:
    """Sharp checkerboard — high Laplacian variance."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if (x // tile + y // tile) % 2 == 0:
                img[y:y+tile, x:x+tile] = 255
    return img


def make_blurred(image: np.ndarray, ksize: int = 201) -> np.ndarray:
    """Heavily blurred version of an image."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def make_flat_grey(height=868, width=1152, value=128) -> np.ndarray:
    """Uniform image — zero spatial frequency, zero variance."""
    return np.full((height, width, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# QualityGate.evaluate
# ---------------------------------------------------------------------------

class TestQualityGateEvaluate:
    def test_sharp_image_passes(self):
        gate = QualityGate(min_laplacian=DEFAULT_MIN_LAPLACIAN)
        result = gate.evaluate(make_checkerboard())
        assert result.passed is True

    def test_blurred_image_fails(self):
        gate = QualityGate(min_laplacian=DEFAULT_MIN_LAPLACIAN)
        sharp = make_checkerboard()
        blurred = make_blurred(sharp)
        result = gate.evaluate(blurred)
        assert result.passed is False

    def test_flat_grey_fails(self):
        gate = QualityGate(min_laplacian=DEFAULT_MIN_LAPLACIAN)
        result = gate.evaluate(make_flat_grey())
        assert result.passed is False

    def test_result_type(self):
        gate = QualityGate()
        result = gate.evaluate(make_checkerboard())
        assert isinstance(result, QualityResult)

    def test_result_fields_populated(self):
        gate = QualityGate(min_laplacian=10.0)
        result = gate.evaluate(make_checkerboard())
        assert result.laplacian_variance > 0
        assert result.threshold == 10.0
        assert isinstance(result.message, str)
        assert isinstance(result.passed, bool)

    def test_pass_message_contains_pass(self):
        gate = QualityGate(min_laplacian=DEFAULT_MIN_LAPLACIAN)
        result = gate.evaluate(make_checkerboard())
        assert "PASS" in result.message

    def test_fail_message_contains_fail(self):
        gate = QualityGate(min_laplacian=DEFAULT_MIN_LAPLACIAN)
        result = gate.evaluate(make_flat_grey())
        assert "FAIL" in result.message

    def test_sharp_has_higher_variance_than_blurred(self):
        gate = QualityGate()
        sharp   = gate.evaluate(make_checkerboard())
        blurred = gate.evaluate(make_blurred(make_checkerboard()))
        assert sharp.laplacian_variance > blurred.laplacian_variance

    def test_custom_threshold_low_accepts_blurred(self):
        gate = QualityGate(min_laplacian=0.0)
        result = gate.evaluate(make_blurred(make_checkerboard()))
        assert result.passed is True

    def test_custom_threshold_high_rejects_sharp(self):
        gate = QualityGate(min_laplacian=1e9)
        result = gate.evaluate(make_checkerboard())
        assert result.passed is False

    def test_variance_is_rounded(self):
        gate = QualityGate()
        result = gate.evaluate(make_checkerboard())
        # Should be rounded to 2 decimal places
        assert result.laplacian_variance == round(result.laplacian_variance, 2)

    def test_works_on_small_image(self):
        """Should not crash on a very small image."""
        gate = QualityGate()
        tiny = make_checkerboard(height=32, width=32, tile=4)
        result = gate.evaluate(tiny)
        assert isinstance(result.passed, bool)

    def test_works_on_non_standard_resolution(self):
        gate = QualityGate()
        img = make_checkerboard(height=480, width=640)
        result = gate.evaluate(img)
        assert result.laplacian_variance > 0


# ---------------------------------------------------------------------------
# QualityGate.evaluate_path
# ---------------------------------------------------------------------------

class TestQualityGateEvaluatePath:
    def test_evaluates_saved_image(self, tmp_path):
        gate = QualityGate()
        img_path = tmp_path / "sharp.png"
        cv2.imwrite(str(img_path), make_checkerboard())
        result = gate.evaluate_path(img_path)
        assert result.passed is True

    def test_missing_file_raises_value_error(self, tmp_path):
        gate = QualityGate()
        with pytest.raises(ValueError, match="Could not read"):
            gate.evaluate_path(tmp_path / "nonexistent.png")


# ---------------------------------------------------------------------------
# Downsample factor
# ---------------------------------------------------------------------------

class TestDownsampleFactor:
    def test_default_downsample_factor(self):
        gate = QualityGate()
        assert gate.downsample_factor == DEFAULT_DOWNSAMPLE_FACTOR

    def test_custom_downsample_factor(self):
        gate = QualityGate(downsample_factor=2)
        assert gate.downsample_factor == 2

    def test_downsample_1_same_as_full_res(self):
        """Factor=1 should still produce a valid result."""
        gate = QualityGate(downsample_factor=1)
        result = gate.evaluate(make_checkerboard(height=64, width=64, tile=8))
        assert result.laplacian_variance > 0
