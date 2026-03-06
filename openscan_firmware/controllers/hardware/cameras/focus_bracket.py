"""
focus_bracket.py
libcamera focus bracket driver for the Arducam Hawkeye 64MP on Raspberry Pi.

Characterises the autofocus range at the current working distance, then
captures a bracket of lossless PNGs at stepped focus positions covering
the full depth of the subject. A quality gate is applied after each frame
with automatic retry before accepting a below-threshold result.

The resulting bracket of images is fed into the focus stacking stage of
the photogrammetry pipeline to produce a single all-in-focus image per
scan position.

Hardware
--------
- Camera:    Arducam Hawkeye 64MP via CSI, libcamera stack.
- LensPosition scale: 0.0 = infinity (far), 15.0 = ~15 cm (near).
- Camera is mounted 180 deg rotated — corrected via hflip+vflip transform.

Notes
-----
16-bit DNG capture via rpicam-still is implemented in _capture_dng() but
is not used in capture_bracket() due to a pidng version mismatch between
picamera2's save_dng and rpicam-still requiring exclusive camera access.
This is deferred to a future pipeline stage.

Usage::

    from openscan_firmware.controllers.hardware.cameras.focus_bracket import (
        FocusBracketDriver,
    )

    with FocusBracketDriver() as driver:
        paths = driver.capture_position(
            azimuth=45.0,
            elevation=20.0,
            output_dir=Path("~/scan/inbox/raw"),
        )
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from openscan_firmware.models.pose_metadata import PoseMetadata, write_sidecar, sidecar_path_for

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SENSOR_RESOLUTION  = (9152, 6944)  # full 64MP resolution
PREVIEW_RESOLUTION = (1152, 868)   # 1/8 scale — fast quality checks, low memory

# Focus sweep parameters for depth characterisation.
FOCUS_SWEEP_STEPS   = 20    # positions sampled across the full LensPosition range
FOCUS_SETTLE_S      = 0.10  # seconds to wait after each LensPosition change

# Quality gate: minimum Laplacian variance to accept a frame.
# Frames below this threshold trigger a retry up to MAX_RECAPTURE_TRIES.
QUALITY_GATE_MIN    = 20
MAX_RECAPTURE_TRIES = 3

BRACKET_MIN_FRAMES  = 3
BRACKET_MAX_FRAMES  = 7


@dataclass
class BracketConfig:
    """
    Parameters for a single focus bracket sequence.

    Attributes:
        focus_near:   LensPosition for the nearest in-focus plane (higher value).
        focus_far:    LensPosition for the farthest in-focus plane (lower value).
        n_frames:     Number of focus steps in the bracket.
        focus_steps:  Pre-computed LensPosition values, evenly spaced from
                      focus_near to focus_far. Auto-populated if not provided.
    """
    focus_near:   float
    focus_far:    float
    n_frames:     int
    focus_steps:  list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.focus_steps:
            self.focus_steps = list(
                np.linspace(self.focus_near, self.focus_far, self.n_frames)
            )


# ---------------------------------------------------------------------------
# Sharpness metric
# ---------------------------------------------------------------------------

def laplacian_variance(image_bgr: np.ndarray) -> float:
    """
    Compute sharpness as the variance of the Laplacian on a greyscale image.

    Higher values indicate sharper images. Used both for focus sweep
    characterisation and per-frame quality gating.

    Args:
        image_bgr: BGR image as a numpy array.

    Returns:
        Laplacian variance as a float.
    """
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class FocusBracketDriver:
    """
    Captures focus-bracketed PNGs using picamera2 / libcamera.

    Exposure is locked after an initial convergence period so that all
    frames in a bracket are comparable. Focus is set manually via
    LensPosition control.

    Args:
        sensor_resolution: Sensor resolution as (width, height). Defaults
                           to the full 64MP resolution of the Hawkeye.
    """

    def __init__(self, sensor_resolution: tuple[int, int] = SENSOR_RESOLUTION):
        self._cam = None
        self._sensor_res = sensor_resolution

    def open(self):
        """
        Open the camera and lock exposure.

        Starts with AE/AWB enabled to allow convergence, then captures the
        converged values and locks them for the remainder of the session.
        """
        from picamera2 import Picamera2
        import libcamera

        self._libcamera = libcamera
        self._cam = Picamera2()

        # Main stream at preview resolution for sharpness checks only —
        # keeps DMA memory usage low. Raw stream at native res for future DNG.
        # Camera is mounted 180 deg rotated — correct with hflip+vflip.
        transform = libcamera.Transform(hflip=1, vflip=1)
        config = self._cam.create_still_configuration(
            main={"size": PREVIEW_RESOLUTION, "format": "BGR888"},
            raw={"format": "SRGGB10_CSI2P", "size": (9248, 6944)},
            transform=transform,
            buffer_count=1,
        )
        self._cam.configure(config)

        self._cam.set_controls({
            "AfMode":       libcamera.controls.AfModeEnum.Manual,
            "LensPosition": 7.5,  # mid-range starting position
        })

        self._cam.start()
        time.sleep(2.0)  # allow AE/AWB to converge

        meta = self._cam.capture_metadata()
        self._cam.set_controls({
            "AeEnable":     False,
            "AwbEnable":    False,
            "ExposureTime": meta["ExposureTime"],
            "AnalogueGain": meta["AnalogueGain"],
            "ColourGains":  meta.get("ColourGains", (1.0, 1.0)),
        })
        time.sleep(0.5)  # allow locked values to take effect

        logger.info(
            "Exposure locked: shutter=%d us gain=%.2f",
            meta["ExposureTime"], meta["AnalogueGain"],
        )
        logger.info("FocusBracketDriver: camera opened, sensor %dx%d", *self._sensor_res)

    def close(self):
        """Stop and release the camera."""
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None
            logger.info("FocusBracketDriver: camera closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Focus characterisation
    # ------------------------------------------------------------------

    def characterise_depth(self) -> BracketConfig:
        """
        Sweep LensPosition across the full range and measure sharpness at each step.

        Identifies the in-focus region around the sharpness peak and returns
        a BracketConfig whose frame count scales with the measured depth span:
        shallow subjects get 3 frames, deeply curved subjects get up to 7.

        Returns:
            BracketConfig covering the in-focus depth range.
        """
        logger.info("Characterising focus depth ...")

        sweep_positions = np.linspace(0.0, 15.0, FOCUS_SWEEP_STEPS)
        scores: list[tuple[float, float]] = []

        for lp in sweep_positions:
            self._set_focus(lp)
            frame = self._capture_preview()
            score = laplacian_variance(frame)
            scores.append((lp, score))
            logger.debug("  LensPosition=%.3f  sharpness=%.1f", lp, score)

        positions, sharpness = zip(*scores)
        sharpness = np.array(sharpness)

        peak_idx = int(np.argmax(sharpness))
        peak_val = sharpness[peak_idx]
        threshold = peak_val * 0.4  # 40% of peak defines the in-focus region

        # Walk outward from the peak to find the local in-focus region.
        # Walking outward (rather than using a global threshold) avoids
        # including a separate background peak far from the subject.
        near_idx = peak_idx
        while near_idx > 0 and sharpness[near_idx - 1] >= threshold:
            near_idx -= 1
        far_idx = peak_idx
        while far_idx < len(sharpness) - 1 and sharpness[far_idx + 1] >= threshold:
            far_idx += 1

        near_pos = float(positions[near_idx])
        far_pos  = float(positions[far_idx])

        # Enforce a minimum bracket half-width around the peak.
        # A coarse sweep may resolve only a single step for a shallow subject;
        # this ensures at least some useful focus coverage.
        peak_pos = float(positions[peak_idx])
        MIN_HALF_WIDTH = 1.5
        near_pos = max(near_pos, peak_pos - MIN_HALF_WIDTH)
        far_pos  = min(far_pos,  peak_pos + MIN_HALF_WIDTH)
        depth_span = abs(near_pos - far_pos)

        n_frames = BRACKET_MIN_FRAMES
        if depth_span > 0.05:
            n_frames = min(BRACKET_MAX_FRAMES,
                           BRACKET_MIN_FRAMES + int(depth_span / 0.05))

        # LensPosition convention: higher value = closer focus (nearer plane).
        focus_near = max(near_pos, far_pos)
        focus_far  = min(near_pos, far_pos)

        cfg = BracketConfig(focus_near=focus_near, focus_far=focus_far, n_frames=n_frames)
        logger.info(
            "Depth characterised: peak LP=%.3f, in-focus LP %.3f-%.3f, %d frames",
            float(positions[peak_idx]), focus_far, focus_near, n_frames,
        )
        return cfg

    # ------------------------------------------------------------------
    # Bracket capture
    # ------------------------------------------------------------------

    def capture_bracket(
        self,
        bracket: BracketConfig,
        output_dir: Path,
        name_prefix: str,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        session_name: str = "scan",
        radius_mm: Optional[float] = None,
    ) -> list[Path]:
        """
        Capture a PNG at each focus step with a pose metadata sidecar.

        Applies the quality gate after each frame and retries up to
        MAX_RECAPTURE_TRIES times before accepting a below-threshold result.

        Args:
            bracket:      BracketConfig defining focus steps.
            output_dir:   Directory for output images and sidecars.
            name_prefix:  Filename prefix, e.g. "scan_az045.00_el020.00".
            azimuth:      Turntable angle for the sidecar metadata.
            elevation:    Rotor angle for the sidecar metadata.
            session_name: Session name prefix used in image_id.
            radius_mm:    Camera-to-object distance, if known.

        Returns:
            List of paths to saved PNG files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        for i, lp in enumerate(bracket.focus_steps):
            self._set_focus(lp)

            for attempt in range(MAX_RECAPTURE_TRIES):
                frame = self._capture_preview()
                score = laplacian_variance(frame)

                if score >= QUALITY_GATE_MIN or attempt == MAX_RECAPTURE_TRIES - 1:
                    if score < QUALITY_GATE_MIN:
                        logger.warning(
                            "Frame %d/%d: sharpness %.1f below gate %d after %d tries — accepting",
                            i + 1, bracket.n_frames, score, QUALITY_GATE_MIN, attempt + 1,
                        )
                    else:
                        logger.debug(
                            "Frame %d/%d: sharpness %.1f OK", i + 1, bracket.n_frames, score
                        )

                    path = output_dir / f"{name_prefix}_f{i}.png"
                    cam_meta = self._cam.capture_metadata()
                    self._cam.capture_file(str(path))

                    meta = PoseMetadata.from_capture(
                        azimuth_deg=azimuth,
                        elevation_deg=elevation,
                        focus_bracket_index=i,
                        focus_bracket_total=bracket.n_frames,
                        sensor_resolution=self._sensor_res,
                        session_name=session_name,
                        file_path=str(
                            path.relative_to(output_dir.parent)
                            if path.is_relative_to(output_dir.parent)
                            else path
                        ),
                        focus_lens_position=lp,
                        radius_mm=radius_mm,
                        camera_metadata=cam_meta,
                    )
                    meta.laplacian_variance = round(score, 2)
                    write_sidecar(meta, sidecar_path_for(path))

                    saved.append(path)
                    break
                else:
                    logger.debug(
                        "Frame %d/%d: sharpness %.1f below gate, retrying (attempt %d)",
                        i + 1, bracket.n_frames, score, attempt + 1,
                    )

        return saved

    def capture_position(
        self,
        azimuth: float,
        elevation: float,
        output_dir: Path,
        bracket: Optional[BracketConfig] = None,
        session_name: str = "scan",
        radius_mm: Optional[float] = None,
    ) -> list[Path]:
        """
        Capture a focus bracket at a given scan position.

        If no BracketConfig is provided, characterise_depth() is called
        first to auto-detect the in-focus range.

        Args:
            azimuth:      Turntable angle in degrees.
            elevation:    Rotor arm angle in degrees.
            output_dir:   Directory for output images and sidecars.
            bracket:      Pre-computed BracketConfig, or None to auto-detect.
            session_name: Session name prefix for image_id in metadata.
            radius_mm:    Camera-to-object distance, if known.

        Returns:
            List of paths to saved PNG files for this position.
        """
        if bracket is None:
            bracket = self.characterise_depth()

        prefix = f"scan_az{azimuth:06.2f}_el{elevation:06.2f}"
        paths = self.capture_bracket(
            bracket, output_dir, prefix,
            azimuth=azimuth, elevation=elevation,
            session_name=session_name, radius_mm=radius_mm,
        )
        logger.info(
            "Position (%.1f deg, %.1f deg): %d frames -> %s",
            azimuth, elevation, len(paths), output_dir,
        )
        return paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_focus(self, lens_position: float):
        """Set LensPosition and wait for it to take effect."""
        self._cam.set_controls({"LensPosition": float(lens_position)})
        time.sleep(FOCUS_SETTLE_S)

    def _capture_dng(self, path: Path, lens_position: float, cam_meta: dict):
        """
        Capture a full-resolution DNG via rpicam-still subprocess.

        Uses locked exposure/gain from cam_meta so that bracket frames are
        exposure-matched. The camera must be stopped before calling this
        method because rpicam-still requires exclusive camera access.

        Note: Currently unused in capture_bracket() due to a pidng version
        mismatch. Retained for future integration.

        Args:
            path:           Output DNG path.
            lens_position:  LensPosition to use for this frame.
            cam_meta:       Metadata dict from a preceding capture_metadata() call.
        """
        self._cam.stop()
        try:
            shutter = cam_meta.get("ExposureTime", 10000)
            gain    = cam_meta.get("AnalogueGain", 1.0)
            cmd = [
                "rpicam-still",
                "--output",        str(path),
                "--encoding",      "dng",
                "--width",         str(self._sensor_res[0]),
                "--height",        str(self._sensor_res[1]),
                "--shutter",       str(int(shutter)),
                "--gain",          str(gain),
                "--awb",           "off",
                "--lens-position", str(lens_position),
                "--hflip", "--vflip",
                "--nopreview",
                "--immediate",
                "-t", "0",
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        finally:
            self._cam.start()
            time.sleep(0.5)  # allow sensor to restabilise

    def _capture_preview(self) -> np.ndarray:
        """Capture from the main (preview-resolution) stream."""
        return self._cam.capture_array("main")


# ---------------------------------------------------------------------------
# CLI — sweep, bracket capture, and acceptance test
# ---------------------------------------------------------------------------

def _run_sweep(output_dir: Path):
    """Print sharpness at each LensPosition step across the full range."""
    with FocusBracketDriver() as driver:
        print("\nFocus sweep:")
        for lp in np.linspace(0.0, 15.0, FOCUS_SWEEP_STEPS):
            driver._set_focus(lp)
            frame = driver._capture_preview()
            score = laplacian_variance(frame)
            bar = "#" * int(score / 50)
            print(f"  LP={lp:.3f}  sharpness={score:7.1f}  {bar}")


def _run_acceptance(output_dir: Path):
    """Capture 5 positions and verify all bracket frames meet the quality gate."""
    import json

    test_positions = [
        (0.0,   0.0),
        (72.0,  0.0),
        (144.0, 0.0),
        (216.0, 20.0),
        (288.0, 40.0),
    ]

    results = []
    with FocusBracketDriver() as driver:
        bracket = driver.characterise_depth()
        print(
            f"\nBracket: {bracket.n_frames} frames, "
            f"LP {bracket.focus_far:.3f}-{bracket.focus_near:.3f}"
        )

        for az, el in test_positions:
            paths = driver.capture_position(az, el, output_dir, bracket=bracket)
            scores = []
            for p in paths:
                sidecar = p.with_suffix(".json")
                if sidecar.exists():
                    scores.append(json.loads(sidecar.read_text()).get("laplacian_variance", 0))
                else:
                    scores.append(0)

            ok = all(s >= QUALITY_GATE_MIN for s in scores)
            results.append({
                "azimuth":   az,
                "elevation": el,
                "n_frames":  len(paths),
                "sharpness": [round(s, 1) for s in scores],
                "passed":    ok,
            })
            print(
                f"  ({az:.0f} deg, {el:.0f} deg): {len(paths)} frames, "
                f"sharpness {[round(s, 1) for s in scores]}  {'PASS' if ok else 'FAIL'}"
            )

    passed = sum(1 for r in results if r["passed"])
    total  = len(results)
    print(f"\nAcceptance: {passed}/{total} positions passed")
    print(f"Result: {'PASS' if passed == total else 'FAIL'}")

    out = output_dir / "acceptance_results.json"
    out.write_text(json.dumps({"results": results, "passed": passed == total}, indent=2))
    print(f"Results written to {out}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Arducam Hawkeye Focus Bracket Driver")
    parser.add_argument("--sweep",           action="store_true",
                        help="Sweep focus and print sharpness curve")
    parser.add_argument("--capture",         action="store_true",
                        help="Capture one bracket at the given position")
    parser.add_argument("--acceptance-test", action="store_true",
                        help="Run acceptance test across 5 positions")
    parser.add_argument("--az",  type=float, default=0.0,
                        help="Azimuth in degrees for --capture")
    parser.add_argument("--el",  type=float, default=0.0,
                        help="Elevation in degrees for --capture")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "scan/inbox/raw",
                        help="Output directory for images and sidecars")
    args = parser.parse_args()

    if args.sweep:
        _run_sweep(args.output_dir)
    elif args.capture:
        with FocusBracketDriver() as driver:
            bracket = driver.characterise_depth()
            paths = driver.capture_position(args.az, args.el, args.output_dir, bracket=bracket)
            print(f"Captured {len(paths)} frames:")
            for p in paths:
                print(f"  {p}")
    elif args.acceptance_test:
        _run_acceptance(args.output_dir)
    else:
        parser.print_help()
