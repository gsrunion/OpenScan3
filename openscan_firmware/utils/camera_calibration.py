"""
camera_calibration.py
OpenCV camera calibration for the Arducam Hawkeye 64MP on Raspberry Pi.

Captures a set of checkerboard images at different orientations, runs
OpenCV calibration, and saves camera intrinsics and distortion coefficients
to a JSON file. The resulting calibration is consumed by the photogrammetry
pipeline for lens distortion correction and as a prior for bundle adjustment.

Target reprojection error: < 0.5 px RMS.

Workflow
--------
**Interactive capture on the Pi** (collect checkerboard images)::

    python -m openscan_firmware.utils.camera_calibration \\
        --capture --output-dir ~/scan/calibration

**Calibrate from existing images** (can run on workstation)::

    python -m openscan_firmware.utils.camera_calibration \\
        --calibrate \\
        --image-dir ~/scan/calibration \\
        --output ~/scan/calibration/calibration_YYYYMMDD.json

**Combined capture and calibrate** (Pi only)::

    python -m openscan_firmware.utils.camera_calibration --capture-and-calibrate

Checkerboard
------------
Use a standard A4 printed checkerboard with 9x6 inner corners and 25 mm
squares. Vary the distance and tilt angle between captures to improve
conditioning of the calibration solve.
"""

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Checkerboard geometry — must match the physical calibration target.
CHECKERBOARD_COLS  = 9     # number of inner corners along the width
CHECKERBOARD_ROWS  = 6     # number of inner corners along the height
SQUARE_SIZE_MM     = 25.0  # physical square size in millimetres

# Capture settings.
MIN_CAPTURES       = 12    # minimum valid frames required for calibration
TARGET_CAPTURES    = 20    # ideal number of frames for a well-conditioned solve
# 1/4 sensor resolution: fast to capture while retaining sufficient detail.
CAPTURE_RESOLUTION = (2312, 1736)


def detect_checkerboard(
    image_bgr: np.ndarray,
    cols: int = CHECKERBOARD_COLS,
    rows: int = CHECKERBOARD_ROWS,
) -> tuple[bool, np.ndarray | None]:
    """
    Detect checkerboard corners in a BGR image and refine to sub-pixel accuracy.

    Args:
        image_bgr: Input image in BGR format.
        cols:      Number of inner corners along the width.
        rows:      Number of inner corners along the height.

    Returns:
        (found, corners) where corners is None if the board was not detected.
    """
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(grey, (cols, rows), None)

    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)

    return found, corners


def calibrate_from_images(
    image_paths: list[Path],
    cols: int = CHECKERBOARD_COLS,
    rows: int = CHECKERBOARD_ROWS,
    square_size_mm: float = SQUARE_SIZE_MM,
) -> dict:
    """
    Run OpenCV camera calibration from a set of checkerboard images.

    Requires at least 4 valid frames; 12-20 is recommended for a stable solve.

    Args:
        image_paths:    Paths to checkerboard images (PNG or JPEG).
        cols:           Inner corner count along the width.
        rows:           Inner corner count along the height.
        square_size_mm: Physical square size in mm.

    Returns:
        Calibration dict containing camera_matrix, dist_coeffs, RMS error,
        and per-image reprojection errors.

    Raises:
        ValueError: If fewer than 4 valid frames are found.
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []
    img_points = []
    valid_images = []
    image_size = None

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("Could not read: %s", path)
            continue

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])

        found, corners = detect_checkerboard(img, cols, rows)
        if found:
            obj_points.append(objp)
            img_points.append(corners)
            valid_images.append(path)
            logger.info("  valid: %s", path.name)
        else:
            logger.warning("  no board: %s", path.name)

    if len(obj_points) < 4:
        raise ValueError(
            f"Only {len(obj_points)} valid images found — need at least 4 for calibration."
        )

    logger.info("Calibrating with %d/%d valid images ...", len(obj_points), len(image_paths))

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    per_image_errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        err = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        per_image_errors.append(round(float(err), 4))

    result = {
        "version":                    "1.0",
        "rms_reprojection_error_px":  round(float(rms), 4),
        "image_size":                 list(image_size),
        "sensor_resolution":          [9152, 6944],
        "capture_resolution":         list(CAPTURE_RESOLUTION),
        "camera_matrix":              camera_matrix.tolist(),
        "dist_coeffs":                dist_coeffs.tolist(),
        "checkerboard": {
            "cols":           cols,
            "rows":           rows,
            "square_size_mm": square_size_mm,
        },
        "n_images_used":    len(obj_points),
        "n_images_total":   len(image_paths),
        "valid_images":     [p.name for p in valid_images],
        "per_image_errors": per_image_errors,
    }

    logger.info(
        "Calibration complete: RMS=%.4f px (%s)",
        rms, "PASS" if rms < 0.5 else "WARN — above 0.5 px target",
    )
    return result


def save_calibration(cal: dict, output_path: Path) -> Path:
    """
    Write a calibration dict to a JSON file.

    Args:
        cal:         Calibration dict from calibrate_from_images().
        output_path: Destination file path. Parent directories are created.

    Returns:
        The path the file was written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cal, indent=2))
    logger.info("Calibration saved: %s", output_path)
    return output_path


class CalibrationCapture:
    """
    Interactive capture session for collecting checkerboard images on the Pi.

    Captures frames automatically every 2 seconds until the target number
    of valid detections is reached. Requires picamera2 and libcamera.

    Args:
        output_dir: Directory where captured images are saved.
        n_target:   Number of valid checkerboard frames to collect.
    """

    def __init__(self, output_dir: Path, n_target: int = TARGET_CAPTURES):
        self.output_dir = Path(output_dir)
        self.n_target = n_target
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> int:
        """
        Run the interactive capture session.

        Captures a frame every 2 seconds, checks for a checkerboard, and
        saves the image if the board is found. Continues until n_target
        valid captures are collected.

        Returns:
            Number of valid frames captured.
        """
        from picamera2 import Picamera2
        import libcamera

        cam = Picamera2()
        transform = libcamera.Transform(hflip=1, vflip=1)
        config = cam.create_still_configuration(
            main={"size": CAPTURE_RESOLUTION, "format": "BGR888"},
            transform=transform,
        )
        cam.configure(config)
        cam.start()
        time.sleep(2.0)  # allow AE/AWB to converge

        captured = 0
        attempt = 0
        print(f"\nCalibration capture — target: {self.n_target} valid frames")
        print("Hold the checkerboard at different angles and distances.")
        print("Capturing automatically every 2 seconds...\n")

        try:
            while captured < self.n_target:
                time.sleep(2.0)
                frame = cam.capture_array("main")
                found, _ = detect_checkerboard(frame)
                attempt += 1

                if found:
                    path = self.output_dir / f"cal_{captured:03d}.png"
                    cv2.imwrite(str(path), frame)
                    captured += 1
                    print(f"  [{captured:2d}/{self.n_target}] found — saved {path.name}")
                else:
                    print(f"  [attempt {attempt}] no checkerboard detected — adjust position")
        finally:
            cam.stop()
            cam.close()

        print(f"\nCapture complete: {captured} frames saved to {self.output_dir}")
        return captured


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from datetime import date

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Camera Calibration — Arducam Hawkeye 64MP")
    parser.add_argument("--capture",               action="store_true",
                        help="Capture calibration images on Pi (requires picamera2)")
    parser.add_argument("--calibrate",             action="store_true",
                        help="Run calibration from existing images")
    parser.add_argument("--capture-and-calibrate", action="store_true",
                        help="Capture then calibrate (Pi only)")
    parser.add_argument("--image-dir",  type=Path,
                        default=Path.home() / "scan/calibration",
                        help="Directory containing calibration images")
    parser.add_argument("--output",     type=Path,
                        default=Path.home() / f"scan/calibration/calibration_{date.today().strftime('%Y%m%d')}.json",
                        help="Output calibration JSON path")
    parser.add_argument("--cols",       type=int,   default=CHECKERBOARD_COLS,
                        help=f"Inner corner count along width (default: {CHECKERBOARD_COLS})")
    parser.add_argument("--rows",       type=int,   default=CHECKERBOARD_ROWS,
                        help=f"Inner corner count along height (default: {CHECKERBOARD_ROWS})")
    parser.add_argument("--square-mm",  type=float, default=SQUARE_SIZE_MM,
                        help=f"Physical square size in mm (default: {SQUARE_SIZE_MM})")
    args = parser.parse_args()

    if args.capture or args.capture_and_calibrate:
        cc = CalibrationCapture(args.image_dir)
        cc.run()

    if args.calibrate or args.capture_and_calibrate:
        image_paths = sorted(args.image_dir.glob("*.png")) + sorted(args.image_dir.glob("*.jpg"))
        if not image_paths:
            print(f"No images found in {args.image_dir}")
            sys.exit(1)
        cal = calibrate_from_images(image_paths, args.cols, args.rows, args.square_mm)
        save_calibration(cal, args.output)
        print(f"\nRMS reprojection error: {cal['rms_reprojection_error_px']} px")
        print(
            f"Target: < 0.5 px  —  "
            f"{'PASS' if cal['rms_reprojection_error_px'] < 0.5 else 'FAIL — recapture with more varied angles'}"
        )
        print(f"Saved: {args.output}")
    elif not args.capture:
        parser.print_help()
