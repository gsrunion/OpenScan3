"""
quality_gate.py
On-device image sharpness gate using Laplacian variance.

Evaluates a captured frame immediately after capture and flags it as
too blurry to use if sharpness falls below a configured threshold. Called
by the focus bracket driver after each capture so that a retry can be
issued before moving to the next focus step.

Algorithm
---------
The Laplacian of a greyscale image highlights edges and fine detail.
Its variance is low for blurry images (few strong gradients) and high
for sharp images. Evaluating on a downsampled version keeps runtime
below 5 ms on Pi 4 hardware at preview resolution.

The default threshold of 20 was chosen empirically as a conservative
floor that rejects motion blur and severe defocus while accepting
legitimately smooth, low-texture subjects. Increase the threshold for
higher sharpness requirements; decrease it for very smooth objects that
genuinely have low spatial frequency content.

Usage::

    from openscan_firmware.utils.quality_gate import QualityGate

    gate = QualityGate(min_laplacian=20)
    result = gate.evaluate(image_bgr)
    if not result.passed:
        # recapture
        pass
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default threshold: conservative floor that rejects motion blur and
# severe defocus. Smooth subjects may require lowering this value.
DEFAULT_MIN_LAPLACIAN = 20

# Evaluate on a 1/4-resolution image for speed. Sharpness ranking is
# preserved at reduced resolution; absolute variance values differ.
DEFAULT_DOWNSAMPLE_FACTOR = 4


@dataclass
class QualityResult:
    """Result of a single quality gate evaluation.

    Attributes:
        passed:             True if sharpness meets the threshold.
        laplacian_variance: Measured variance of the Laplacian.
        threshold:          The minimum threshold that was applied.
        message:            Human-readable PASS/FAIL summary.
    """
    passed:             bool
    laplacian_variance: float
    threshold:          float
    message:            str


class QualityGate:
    """
    Evaluates image sharpness using Laplacian variance.

    Designed to run between captures without measurably slowing the scan
    cycle. Evaluation is performed on a downsampled greyscale version of
    the frame.

    Args:
        min_laplacian:     Minimum Laplacian variance to pass. Frames below
                           this threshold are considered too blurry.
        downsample_factor: Factor by which to reduce image resolution before
                           computing sharpness. Higher values are faster but
                           less sensitive to fine-detail blur.
    """

    def __init__(
        self,
        min_laplacian: float = DEFAULT_MIN_LAPLACIAN,
        downsample_factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    ):
        self.min_laplacian = min_laplacian
        self.downsample_factor = downsample_factor

    def evaluate(self, image_bgr: np.ndarray) -> QualityResult:
        """
        Evaluate the sharpness of a BGR image.

        Args:
            image_bgr: Image as a numpy array in BGR format (any resolution).

        Returns:
            QualityResult with pass/fail verdict and measured variance.
        """
        h, w = image_bgr.shape[:2]
        small = cv2.resize(
            image_bgr,
            (w // self.downsample_factor, h // self.downsample_factor),
            interpolation=cv2.INTER_AREA,
        )

        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(grey, cv2.CV_64F).var())

        passed = variance >= self.min_laplacian
        msg = (
            f"PASS: sharpness={variance:.1f} >= {self.min_laplacian}"
            if passed else
            f"FAIL: sharpness={variance:.1f} < {self.min_laplacian}"
        )

        logger.debug(msg)
        return QualityResult(
            passed=passed,
            laplacian_variance=round(variance, 2),
            threshold=self.min_laplacian,
            message=msg,
        )

    def evaluate_path(self, image_path: Path) -> QualityResult:
        """
        Load an image from disk and evaluate its sharpness.

        Args:
            image_path: Path to a BGR-compatible image file.

        Returns:
            QualityResult for the loaded image.

        Raises:
            ValueError: If the image cannot be read.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.evaluate(img)


# ---------------------------------------------------------------------------
# CLI — acceptance test
# ---------------------------------------------------------------------------

def _run_acceptance_test() -> bool:
    """
    Verify the quality gate using synthetic images (no camera required).

    Tests:
        1. Sharp checkerboard — must PASS.
        2. Heavily blurred image — must FAIL.
        3. Uniform grey image (no edges) — must FAIL.
        4. Low-noise image — reported for reference (threshold-dependent).
    """
    gate = QualityGate(min_laplacian=DEFAULT_MIN_LAPLACIAN)
    errors = []

    # Test 1: sharp checkerboard — must PASS
    sharp = np.zeros((868, 1152, 3), dtype=np.uint8)
    tile = 32
    for y in range(0, 868, tile):
        for x in range(0, 1152, tile):
            if (x // tile + y // tile) % 2 == 0:
                sharp[y:y+tile, x:x+tile] = 255
    result = gate.evaluate(sharp)
    if not result.passed:
        errors.append(f"Sharp image FAILED gate: sharpness={result.laplacian_variance:.1f}")
    else:
        print(f"  Sharp image:   PASS  (sharpness={result.laplacian_variance:.1f})")

    # Test 2: heavily blurred image — must FAIL
    blurred = cv2.GaussianBlur(sharp, (201, 201), 0)
    result = gate.evaluate(blurred)
    if result.passed:
        errors.append(
            f"Blurred image PASSED gate (should fail): sharpness={result.laplacian_variance:.1f}"
        )
    else:
        print(f"  Blurred image: PASS (correctly rejected, sharpness={result.laplacian_variance:.1f})")

    # Test 3: uniform grey — must FAIL (no edges)
    grey_flat = np.full((868, 1152, 3), 128, dtype=np.uint8)
    result = gate.evaluate(grey_flat)
    if result.passed:
        errors.append(
            f"Flat grey PASSED gate (should fail): sharpness={result.laplacian_variance:.1f}"
        )
    else:
        print(f"  Flat grey:     PASS (correctly rejected, sharpness={result.laplacian_variance:.1f})")

    # Test 4: low-noise image — informational (result depends on threshold)
    noise = np.random.normal(128, 5, (868, 1152)).clip(0, 255).astype(np.uint8)
    noise_bgr = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    result = gate.evaluate(noise_bgr)
    print(
        f"  Noise image:   sharpness={result.laplacian_variance:.1f} "
        f"({'PASS' if result.passed else 'FAIL — below gate, expected for low noise'})"
    )

    passed = len(errors) == 0
    print(f"\nAcceptance: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
    return passed


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="On-device Image Quality Gate")
    parser.add_argument("--acceptance-test", action="store_true",
                        help="Run synthetic acceptance test (no camera required)")
    parser.add_argument("--evaluate", type=Path,
                        help="Evaluate a specific image file")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MIN_LAPLACIAN,
                        help=f"Minimum Laplacian variance threshold (default: {DEFAULT_MIN_LAPLACIAN})")
    args = parser.parse_args()

    if args.acceptance_test:
        ok = _run_acceptance_test()
        raise SystemExit(0 if ok else 1)
    elif args.evaluate:
        gate = QualityGate(min_laplacian=args.threshold)
        result = gate.evaluate_path(args.evaluate)
        print(result.message)
        raise SystemExit(0 if result.passed else 1)
    else:
        parser.print_help()
