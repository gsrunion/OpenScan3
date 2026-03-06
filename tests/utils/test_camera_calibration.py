import json
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch

from openscan_firmware.utils.camera_calibration import (
    CHECKERBOARD_COLS,
    CHECKERBOARD_ROWS,
    SQUARE_SIZE_MM,
    detect_checkerboard,
    calibrate_from_images,
    save_calibration,
)

# ---------------------------------------------------------------------------
# Synthetic checkerboard image helpers
# ---------------------------------------------------------------------------

def render_checkerboard_image(
    cols: int = CHECKERBOARD_COLS,
    rows: int = CHECKERBOARD_ROWS,
    square_px: int = 40,
    border_px: int = 20,
) -> np.ndarray:
    """
    Render a synthetic checkerboard image that OpenCV can detect corners in.
    The board has (cols+1) x (rows+1) squares so there are cols x rows inner corners.
    """
    board_w = (cols + 1) * square_px + 2 * border_px
    board_h = (rows + 1) * square_px + 2 * border_px
    img = np.ones((board_h, board_w, 3), dtype=np.uint8) * 200  # light grey background

    for row in range(rows + 1):
        for col in range(cols + 1):
            if (row + col) % 2 == 0:
                x0 = border_px + col * square_px
                y0 = border_px + row * square_px
                img[y0:y0+square_px, x0:x0+square_px] = 0  # black square

    return img


def save_checkerboard_images(tmp_path: Path, n: int = 5, **kwargs) -> list[Path]:
    """Save n synthetic checkerboard PNGs to tmp_path."""
    paths = []
    for i in range(n):
        img = render_checkerboard_image(**kwargs)
        p = tmp_path / f"cal_{i:03d}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# detect_checkerboard
# ---------------------------------------------------------------------------

class TestDetectCheckerboard:
    def test_finds_board_in_synthetic_image(self):
        img = render_checkerboard_image()
        found, corners = detect_checkerboard(img)
        assert found is True
        assert corners is not None

    def test_corners_shape(self):
        img = render_checkerboard_image()
        found, corners = detect_checkerboard(img)
        assert found
        n_inner = CHECKERBOARD_COLS * CHECKERBOARD_ROWS
        assert corners.shape[0] == n_inner

    def test_no_board_returns_false(self):
        blank = np.full((400, 600, 3), 128, dtype=np.uint8)
        found, corners = detect_checkerboard(blank)
        assert found is False
        assert corners is None

    def test_custom_grid_size(self):
        img = render_checkerboard_image(cols=6, rows=4)
        found, corners = detect_checkerboard(img, cols=6, rows=4)
        assert found is True
        assert corners.shape[0] == 6 * 4


# ---------------------------------------------------------------------------
# calibrate_from_images
# ---------------------------------------------------------------------------

class TestCalibrateFromImages:
    def test_raises_with_too_few_valid_images(self, tmp_path):
        """Fewer than 4 detectable images should raise ValueError."""
        blank = np.full((400, 600, 3), 128, dtype=np.uint8)
        paths = []
        for i in range(3):
            p = tmp_path / f"blank_{i}.png"
            cv2.imwrite(str(p), blank)
            paths.append(p)
        with pytest.raises(ValueError, match="need at least 4"):
            calibrate_from_images(paths)

    def test_raises_with_unreadable_files(self, tmp_path):
        """All-corrupt files should raise ValueError."""
        paths = []
        for i in range(5):
            p = tmp_path / f"corrupt_{i}.png"
            p.write_bytes(b"not an image")
            paths.append(p)
        with pytest.raises(ValueError):
            calibrate_from_images(paths)

    def test_returns_dict_with_required_keys(self, tmp_path):
        paths = save_checkerboard_images(tmp_path, n=6)
        result = calibrate_from_images(paths)
        for key in ["camera_matrix", "dist_coeffs", "rms_reprojection_error_px",
                    "image_size", "n_images_used", "n_images_total", "valid_images"]:
            assert key in result, f"Missing key: {key}"

    def test_camera_matrix_shape(self, tmp_path):
        paths = save_checkerboard_images(tmp_path, n=6)
        result = calibrate_from_images(paths)
        cm = result["camera_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_rms_is_float(self, tmp_path):
        paths = save_checkerboard_images(tmp_path, n=6)
        result = calibrate_from_images(paths)
        assert isinstance(result["rms_reprojection_error_px"], float)

    def test_n_images_used_matches_valid_count(self, tmp_path):
        paths = save_checkerboard_images(tmp_path, n=6)
        result = calibrate_from_images(paths)
        assert result["n_images_used"] == len(result["valid_images"])
        assert result["n_images_total"] == 6

    def test_per_image_errors_length(self, tmp_path):
        paths = save_checkerboard_images(tmp_path, n=6)
        result = calibrate_from_images(paths)
        assert len(result["per_image_errors"]) == result["n_images_used"]

    def test_custom_square_size(self, tmp_path):
        paths = save_checkerboard_images(tmp_path, n=6)
        result = calibrate_from_images(paths, square_size_mm=30.0)
        assert result["rms_reprojection_error_px"] is not None

    def test_skips_unreadable_files_gracefully(self, tmp_path):
        """Unreadable files should be skipped; valid ones still calibrate."""
        good_paths = save_checkerboard_images(tmp_path, n=5)
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"garbage")
        all_paths = good_paths + [bad]
        result = calibrate_from_images(all_paths)
        assert result["n_images_total"] == 6
        assert result["n_images_used"] == 5

    def test_checkerboard_metadata_in_result(self, tmp_path):
        paths = save_checkerboard_images(tmp_path, n=6)
        result = calibrate_from_images(paths)
        cb = result["checkerboard"]
        assert cb["cols"]           == CHECKERBOARD_COLS
        assert cb["rows"]           == CHECKERBOARD_ROWS
        assert cb["square_size_mm"] == SQUARE_SIZE_MM


# ---------------------------------------------------------------------------
# save_calibration
# ---------------------------------------------------------------------------

class TestSaveCalibration:
    def test_creates_file(self, tmp_path):
        cal = {"camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               "dist_coeffs": [[0.0]],
               "rms_reprojection_error_px": 0.3}
        path = tmp_path / "calibration.json"
        save_calibration(cal, path)
        assert path.exists()

    def test_output_is_valid_json(self, tmp_path):
        cal = {"key": "value"}
        path = tmp_path / "cal.json"
        save_calibration(cal, path)
        data = json.loads(path.read_text())
        assert data == cal

    def test_creates_parent_directories(self, tmp_path):
        cal = {"key": "value"}
        path = tmp_path / "nested" / "dir" / "cal.json"
        save_calibration(cal, path)
        assert path.exists()

    def test_returns_path(self, tmp_path):
        cal = {}
        path = tmp_path / "cal.json"
        result = save_calibration(cal, path)
        assert result == path
