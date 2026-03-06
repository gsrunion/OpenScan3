import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from openscan_firmware.controllers.hardware.cameras.focus_bracket import (
    BracketConfig,
    laplacian_variance,
    QUALITY_GATE_MIN,
    MAX_RECAPTURE_TRIES,
    BRACKET_MIN_FRAMES,
    BRACKET_MAX_FRAMES,
    FOCUS_SWEEP_STEPS,
)

# ---------------------------------------------------------------------------
# BracketConfig
# ---------------------------------------------------------------------------

class TestBracketConfig:
    def test_focus_steps_auto_populated(self):
        cfg = BracketConfig(focus_near=10.0, focus_far=5.0, n_frames=5)
        assert len(cfg.focus_steps) == 5

    def test_focus_steps_start_at_near(self):
        cfg = BracketConfig(focus_near=10.0, focus_far=5.0, n_frames=3)
        assert cfg.focus_steps[0] == pytest.approx(10.0)

    def test_focus_steps_end_at_far(self):
        cfg = BracketConfig(focus_near=10.0, focus_far=5.0, n_frames=3)
        assert cfg.focus_steps[-1] == pytest.approx(5.0)

    def test_focus_steps_evenly_spaced(self):
        cfg = BracketConfig(focus_near=10.0, focus_far=0.0, n_frames=5)
        diffs = [cfg.focus_steps[i+1] - cfg.focus_steps[i]
                 for i in range(len(cfg.focus_steps) - 1)]
        assert all(abs(d - diffs[0]) < 1e-6 for d in diffs)

    def test_explicit_focus_steps_not_overwritten(self):
        steps = [1.0, 2.0, 3.0]
        cfg = BracketConfig(focus_near=5.0, focus_far=0.0, n_frames=3, focus_steps=steps)
        assert cfg.focus_steps == steps

    def test_single_frame_bracket(self):
        cfg = BracketConfig(focus_near=7.5, focus_far=7.5, n_frames=1)
        assert len(cfg.focus_steps) == 1
        assert cfg.focus_steps[0] == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# laplacian_variance
# ---------------------------------------------------------------------------

class TestLaplacianVariance:
    def test_sharp_image_high_variance(self):
        sharp = np.zeros((200, 200, 3), dtype=np.uint8)
        tile = 10
        for y in range(0, 200, tile):
            for x in range(0, 200, tile):
                if (x // tile + y // tile) % 2 == 0:
                    sharp[y:y+tile, x:x+tile] = 255
        assert laplacian_variance(sharp) > 100

    def test_flat_image_low_variance(self):
        flat = np.full((200, 200, 3), 128, dtype=np.uint8)
        assert laplacian_variance(flat) < 1.0

    def test_returns_float(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        assert isinstance(laplacian_variance(img), float)

    def test_blurred_lower_than_sharp(self):
        import cv2
        sharp = np.zeros((200, 200, 3), dtype=np.uint8)
        for y in range(0, 200, 10):
            for x in range(0, 200, 10):
                if (x // 10 + y // 10) % 2 == 0:
                    sharp[y:y+10, x:x+10] = 255
        blurred = cv2.GaussianBlur(sharp, (51, 51), 0)
        assert laplacian_variance(sharp) > laplacian_variance(blurred)


# ---------------------------------------------------------------------------
# FocusBracketDriver — mocked unit tests
# ---------------------------------------------------------------------------

def make_driver():
    """Return a FocusBracketDriver with picamera2/libcamera fully mocked."""
    from openscan_firmware.controllers.hardware.cameras.focus_bracket import FocusBracketDriver

    driver = FocusBracketDriver.__new__(FocusBracketDriver)
    driver._sensor_res = (9152, 6944)
    driver._cam = MagicMock()
    driver._libcamera = MagicMock()
    return driver


class TestCharacteriseDepth:
    def _make_sharpness_curve(self, peak_idx: int, n: int = FOCUS_SWEEP_STEPS) -> list[float]:
        """Gaussian-shaped sharpness curve peaking at peak_idx."""
        return [
            1000 * np.exp(-0.5 * ((i - peak_idx) / (n / 6)) ** 2)
            for i in range(n)
        ]

    def test_returns_bracket_config(self):
        from openscan_firmware.controllers.hardware.cameras.focus_bracket import FocusBracketDriver
        driver = make_driver()

        sharpness = self._make_sharpness_curve(peak_idx=10)
        idx = 0

        def fake_capture_preview():
            nonlocal idx
            # Return image whose Laplacian variance matches the sharpness curve
            s = sharpness[idx % len(sharpness)]
            idx += 1
            # Make a synthetic image with approximately the right variance
            noise_std = max(0.1, s ** 0.5 / 10)
            img = np.random.normal(128, noise_std, (100, 100, 3)).clip(0, 255).astype(np.uint8)
            return img

        driver._capture_preview = fake_capture_preview
        driver._set_focus = MagicMock()

        result = driver.characterise_depth()

        assert isinstance(result, BracketConfig)
        assert BRACKET_MIN_FRAMES <= result.n_frames <= BRACKET_MAX_FRAMES
        assert len(result.focus_steps) == result.n_frames

    def test_set_focus_called_for_each_sweep_step(self):
        driver = make_driver()
        driver._capture_preview = MagicMock(
            return_value=np.full((100, 100, 3), 128, dtype=np.uint8)
        )
        driver._set_focus = MagicMock()

        driver.characterise_depth()

        assert driver._set_focus.call_count == FOCUS_SWEEP_STEPS


class TestCaptureBracket:
    def _make_sharp_frame(self) -> np.ndarray:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tile = 5
        for y in range(0, 100, tile):
            for x in range(0, 100, tile):
                if (x // tile + y // tile) % 2 == 0:
                    img[y:y+tile, x:x+tile] = 255
        return img

    def _make_blurry_frame(self) -> np.ndarray:
        return np.full((100, 100, 3), 128, dtype=np.uint8)

    def test_saves_one_file_per_focus_step(self, tmp_path):
        driver = make_driver()
        driver._capture_preview = MagicMock(return_value=self._make_sharp_frame())
        driver._set_focus = MagicMock()
        driver._cam.capture_metadata.return_value = {
            "ExposureTime": 10000, "AnalogueGain": 1.0
        }
        driver._cam.capture_file = MagicMock(
            side_effect=lambda p: Path(p).touch()
        )

        bracket = BracketConfig(focus_near=10.0, focus_far=5.0, n_frames=3)
        paths = driver.capture_bracket(bracket, tmp_path, "test_frame",
                                       azimuth=45.0, elevation=20.0)

        assert len(paths) == 3

    def test_sidecar_written_alongside_image(self, tmp_path):
        driver = make_driver()
        driver._capture_preview = MagicMock(return_value=self._make_sharp_frame())
        driver._set_focus = MagicMock()
        driver._cam.capture_metadata.return_value = {
            "ExposureTime": 10000, "AnalogueGain": 1.0
        }
        driver._cam.capture_file = MagicMock(
            side_effect=lambda p: Path(p).touch()
        )

        bracket = BracketConfig(focus_near=10.0, focus_far=5.0, n_frames=2)
        paths = driver.capture_bracket(bracket, tmp_path, "frame",
                                       azimuth=0.0, elevation=0.0)

        for img_path in paths:
            sidecar = img_path.with_suffix(".json")
            assert sidecar.exists(), f"No sidecar for {img_path}"

    def test_sidecar_contains_pose_metadata(self, tmp_path):
        driver = make_driver()
        driver._capture_preview = MagicMock(return_value=self._make_sharp_frame())
        driver._set_focus = MagicMock()
        driver._cam.capture_metadata.return_value = {"ExposureTime": 5000, "AnalogueGain": 1.5}
        driver._cam.capture_file = MagicMock(
            side_effect=lambda p: Path(p).touch()
        )

        bracket = BracketConfig(focus_near=8.0, focus_far=6.0, n_frames=1)
        paths = driver.capture_bracket(bracket, tmp_path, "frame",
                                       azimuth=90.0, elevation=30.0)

        sidecar = json.loads(paths[0].with_suffix(".json").read_text())
        assert sidecar["azimuth_deg"]   == pytest.approx(90.0)
        assert sidecar["elevation_deg"] == pytest.approx(30.0)

    def test_retries_when_below_quality_gate(self, tmp_path):
        driver = make_driver()

        call_count = 0

        def alternating_quality():
            nonlocal call_count
            call_count += 1
            # First call per step: blurry; second: sharp
            if call_count % 2 == 1:
                return self._make_blurry_frame()
            return self._make_sharp_frame()

        driver._capture_preview = alternating_quality
        driver._set_focus = MagicMock()
        driver._cam.capture_metadata.return_value = {"ExposureTime": 5000, "AnalogueGain": 1.0}
        driver._cam.capture_file = MagicMock(
            side_effect=lambda p: Path(p).touch()
        )

        bracket = BracketConfig(focus_near=10.0, focus_far=5.0, n_frames=1)
        paths = driver.capture_bracket(bracket, tmp_path, "frame",
                                       azimuth=0.0, elevation=0.0)

        # capture_preview should have been called more than once for the single frame
        assert call_count >= 2
        assert len(paths) == 1

    def test_accepts_below_gate_after_max_retries(self, tmp_path):
        """After MAX_RECAPTURE_TRIES, a below-threshold frame should still be saved."""
        driver = make_driver()
        driver._capture_preview = MagicMock(return_value=self._make_blurry_frame())
        driver._set_focus = MagicMock()
        driver._cam.capture_metadata.return_value = {"ExposureTime": 5000, "AnalogueGain": 1.0}
        driver._cam.capture_file = MagicMock(
            side_effect=lambda p: Path(p).touch()
        )

        bracket = BracketConfig(focus_near=10.0, focus_far=5.0, n_frames=1)
        paths = driver.capture_bracket(bracket, tmp_path, "frame",
                                       azimuth=0.0, elevation=0.0)

        assert len(paths) == 1
        assert driver._capture_preview.call_count == MAX_RECAPTURE_TRIES

    def test_set_focus_called_for_each_step(self, tmp_path):
        driver = make_driver()
        driver._capture_preview = MagicMock(return_value=self._make_sharp_frame())
        driver._set_focus = MagicMock()
        driver._cam.capture_metadata.return_value = {"ExposureTime": 5000, "AnalogueGain": 1.0}
        driver._cam.capture_file = MagicMock(
            side_effect=lambda p: Path(p).touch()
        )

        n = 4
        bracket = BracketConfig(focus_near=10.0, focus_far=5.0, n_frames=n)
        driver.capture_bracket(bracket, tmp_path, "frame", azimuth=0.0, elevation=0.0)

        assert driver._set_focus.call_count >= n

    def test_output_dir_created_if_missing(self, tmp_path):
        driver = make_driver()
        driver._capture_preview = MagicMock(return_value=self._make_sharp_frame())
        driver._set_focus = MagicMock()
        driver._cam.capture_metadata.return_value = {"ExposureTime": 5000, "AnalogueGain": 1.0}
        driver._cam.capture_file = MagicMock(
            side_effect=lambda p: Path(p).touch()
        )

        new_dir = tmp_path / "new" / "nested" / "dir"
        bracket = BracketConfig(focus_near=10.0, focus_far=5.0, n_frames=1)
        driver.capture_bracket(bracket, new_dir, "frame", azimuth=0.0, elevation=0.0)

        assert new_dir.exists()


class TestCapturePosition:
    def test_calls_characterise_depth_when_no_bracket(self):
        from openscan_firmware.controllers.hardware.cameras.focus_bracket import FocusBracketDriver
        driver = make_driver()
        driver.characterise_depth = MagicMock(
            return_value=BracketConfig(focus_near=10.0, focus_far=5.0, n_frames=2)
        )
        driver.capture_bracket = MagicMock(return_value=[])

        driver.capture_position(45.0, 20.0, Path("/tmp/scan"))

        driver.characterise_depth.assert_called_once()

    def test_skips_characterise_depth_when_bracket_provided(self):
        driver = make_driver()
        driver.characterise_depth = MagicMock()
        driver.capture_bracket = MagicMock(return_value=[])

        bracket = BracketConfig(focus_near=10.0, focus_far=5.0, n_frames=3)
        driver.capture_position(45.0, 20.0, Path("/tmp/scan"), bracket=bracket)

        driver.characterise_depth.assert_not_called()
