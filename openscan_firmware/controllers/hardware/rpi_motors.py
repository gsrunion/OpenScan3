"""
rpi_motors.py
GPIO stepper motor controller for OpenScan Classic on Raspberry Pi.

Provides direct RPi.GPIO-based stepper control as an alternative to the
gpiozero-based MotorController in motors.py. Intended for cases where
tighter control over step timing and acceleration is required.

Public interface (move_to, get_position, execute_scan_plan) is compatible
with the REST-based transport so that orchestration code above this layer
requires no changes when switching transport backends.

Hardware target: OpenScan Classic Green Shield, Pi BCM GPIO numbering.

.. warning::
    GPIO pin assignments below were set during initial development and have
    not been verified against live hardware. The architecture documentation
    (docs/ARCHITECTURE.md §8.1) specifies different pin numbers:

        Architecture doc:  Turntable STEP=21 DIR=20 EN=16
                           Rotor     STEP=26 DIR=19 EN=13

        This file (coded): Turntable STEP=11 DIR=9  EN=22
                           Rotor     STEP=6  DIR=5  EN=23

    Reconcile against the physical PCB before running on hardware.
    See also: https://openscan-org.github.io/OpenScan-Doc/hardware/PCBs/

Motor specs (OpenScan Classic Green Shield):
    Turntable (azimuth theta):   3200 steps/rev  (200 full x 1/16 micro)
    Rotor arm (elevation phi):  17067 steps/rev  (3200 motor x ~5.33 gear ratio, max 140 deg)

Settle delay: 300 ms after any move before capture (see ARCHITECTURE.md §8.2).

Usage::

    async with OpenScanController() as ctrl:
        await ctrl.move_to(azimuth=45.0, elevation=20.0)

Dry-run (no hardware)::

    async with OpenScanController(dry_run=True) as ctrl:
        await ctrl.move_to(45.0, 20.0)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Motor configuration
# ---------------------------------------------------------------------------

@dataclass
class MotorConfig:
    """Pin and motion parameters for a single stepper motor axis."""
    step_pin:         int
    dir_pin:          int
    en_pin:           int
    steps_per_rot:    int    # full steps for one 360 degree rotation
    accel_ramp:       int    # number of steps over which to ramp speed
    step_delay:       float  # base delay between steps at full speed (seconds)
    direction:        int    # 1 or -1, corrects physical wiring direction


# NOTE: Verify these pin numbers against the physical PCB before use.
# See module warning above.
TURNTABLE_CFG = MotorConfig(
    step_pin=11, dir_pin=9, en_pin=22,
    steps_per_rot=3200, accel_ramp=200, step_delay=0.0001, direction=1,
)

ROTOR_CFG = MotorConfig(
    step_pin=6, dir_pin=5, en_pin=23,
    steps_per_rot=17067, accel_ramp=200, step_delay=0.0001, direction=1,
)

SETTLE_DELAY_S = 0.30  # seconds to wait after any move before capture


# ---------------------------------------------------------------------------
# Low-level stepper driver
# ---------------------------------------------------------------------------

class StepperMotor:
    """
    Drives a single stepper motor via GPIO STEP/DIR/EN pins.

    Tracks current position in steps (relative to home = 0).
    Uses a trapezoidal acceleration ramp matching the OpenScan3 firmware
    acceleration logic.

    Args:
        cfg:      Motor pin and motion configuration.
        gpio:     RPi.GPIO module (or compatible mock for dry-run).
        dry_run:  If True, simulate movement without touching GPIO.
    """

    def __init__(self, cfg: MotorConfig, gpio, dry_run: bool = False):
        self.cfg = cfg
        self._gpio = gpio
        self.dry_run = dry_run
        self._position_steps: int = 0

        if not dry_run:
            gpio.setup(cfg.step_pin, gpio.OUT, initial=gpio.LOW)
            gpio.setup(cfg.dir_pin,  gpio.OUT, initial=gpio.LOW)
            gpio.setup(cfg.en_pin,   gpio.OUT, initial=gpio.HIGH)  # HIGH = disabled

    def _deg_to_steps(self, degrees: float) -> int:
        return round(degrees * self.cfg.steps_per_rot / 360.0)

    def _steps_to_deg(self, steps: int) -> float:
        return steps * 360.0 / self.cfg.steps_per_rot

    @property
    def position_deg(self) -> float:
        return self._steps_to_deg(self._position_steps)

    def _enable(self):
        if not self.dry_run:
            self._gpio.output(self.cfg.en_pin, self._gpio.LOW)   # LOW = enabled

    def _disable(self):
        if not self.dry_run:
            self._gpio.output(self.cfg.en_pin, self._gpio.HIGH)  # HIGH = disabled

    def _step_delay_for(self, step_index: int, total_steps: int) -> float:
        """
        Return the inter-step delay for a given step index using a trapezoidal
        ramp: slow at start and end, full speed in the middle.

        Delay ranges from 4x base_delay at the endpoints down to 1x base_delay
        at full speed.
        """
        ramp = self.cfg.accel_ramp
        base = self.cfg.step_delay

        ramp_progress = min(step_index, total_steps - step_index, ramp)
        if ramp_progress <= 0:
            return base * 4.0

        factor = 1.0 + 3.0 * (1.0 - ramp_progress / ramp)
        return base * factor

    async def move_to_deg(self, target_deg: float) -> float:
        """
        Move to an absolute position in degrees.

        The step loop runs in a thread executor so the asyncio event loop
        remains unblocked during physical movement.

        Args:
            target_deg: Target position in degrees.

        Returns:
            Confirmed position in degrees after the move.
        """
        target_steps = self._deg_to_steps(target_deg)
        delta = target_steps - self._position_steps

        if delta == 0:
            return self.position_deg

        logger.debug(
            "Motor STEP=%d: %.2f deg -> %.2f deg (%+d steps)",
            self.cfg.step_pin, self.position_deg, target_deg, delta,
        )

        if self.dry_run:
            await asyncio.sleep(min(abs(delta) * 0.00005, 0.5))
            self._position_steps = target_steps
            return self.position_deg

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._blocking_move, delta)
        return self.position_deg

    def _blocking_move(self, delta: int):
        """
        Blocking step loop — runs in a thread executor.

        Keeps GPIO timing tight without holding the event loop. Direction is
        corrected by cfg.direction to account for physical wiring.
        """
        gpio = self._gpio
        cfg = self.cfg
        total = abs(delta)
        direction = (1 if delta > 0 else -1) * cfg.direction

        gpio.output(cfg.dir_pin, gpio.HIGH if direction > 0 else gpio.LOW)

        self._enable()
        try:
            for i in range(total):
                delay = self._step_delay_for(i, total)
                gpio.output(cfg.step_pin, gpio.HIGH)
                time.sleep(delay / 2)
                gpio.output(cfg.step_pin, gpio.LOW)
                time.sleep(delay / 2)
            self._position_steps += delta
        finally:
            self._disable()

    def reset_position(self):
        """Reset the step counter to zero. Call after physical homing."""
        self._position_steps = 0


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Confirmed two-axis position after a move."""
    azimuth_deg:   float
    elevation_deg: float

    def __str__(self):
        return f"theta={self.azimuth_deg:.2f}deg phi={self.elevation_deg:.2f}deg"


# ---------------------------------------------------------------------------
# High-level two-axis controller
# ---------------------------------------------------------------------------

class OpenScanController:
    """
    Async controller for the OpenScan Classic turntable via Raspberry Pi GPIO.

    Manages both motor axes and exposes the same public interface as the
    REST-based transport, so orchestration code above this layer is unchanged.

    Axes:
        Azimuth  (turntable): 0-360 deg  — 3200 steps/rev
        Elevation (rotor):    0-140 deg  — 17067 steps/rev (geared)

    Args:
        turntable_cfg:  Pin/motion config for the turntable axis.
        rotor_cfg:      Pin/motion config for the rotor arm axis.
        settle_delay_s: Seconds to wait after a move before capture.
        dry_run:        If True, simulate without GPIO access.
    """

    def __init__(
        self,
        turntable_cfg: MotorConfig = TURNTABLE_CFG,
        rotor_cfg:     MotorConfig = ROTOR_CFG,
        settle_delay_s: float = SETTLE_DELAY_S,
        dry_run: bool = False,
    ):
        self.settle_delay_s = settle_delay_s
        self.dry_run = dry_run
        self._gpio = None
        self._turntable: Optional[StepperMotor] = None
        self._rotor:     Optional[StepperMotor] = None
        self._turntable_cfg = turntable_cfg
        self._rotor_cfg     = rotor_cfg

        if dry_run:
            logger.info("OpenScanController: DRY-RUN mode — no GPIO access")

    async def __aenter__(self):
        if not self.dry_run:
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                self._gpio = GPIO
            except ImportError:
                raise RuntimeError(
                    "RPi.GPIO not available. "
                    "Install with: pip install RPi.GPIO  "
                    "Or use dry_run=True for offline testing."
                )
        self._turntable = StepperMotor(self._turntable_cfg, self._gpio, self.dry_run)
        self._rotor     = StepperMotor(self._rotor_cfg,     self._gpio, self.dry_run)
        return self

    async def __aexit__(self, *_):
        if self._gpio is not None:
            self._gpio.cleanup()

    async def move_to(self, azimuth: float, elevation: float) -> Position:
        """
        Move both axes concurrently to (azimuth, elevation).

        Applies the configured settle delay after both motors have stopped
        before returning, giving the rig time to dampen vibration.

        Args:
            azimuth:   Target turntable angle in degrees (0-360).
            elevation: Target rotor angle in degrees (0-140).

        Returns:
            Confirmed Position after the move and settle delay.
        """
        logger.info("Moving to theta=%.1f deg phi=%.1f deg", azimuth, elevation)

        await asyncio.gather(
            self._turntable.move_to_deg(azimuth),
            self._rotor.move_to_deg(elevation),
        )

        await asyncio.sleep(self.settle_delay_s)

        pos = await self.get_position()
        logger.info("Confirmed: %s", pos)

        az_err = abs(pos.azimuth_deg - azimuth)
        el_err = abs(pos.elevation_deg - elevation)
        if az_err > 0.5 or el_err > 0.5:
            logger.warning(
                "Position error exceeds +/-0.5 deg: az_err=%.3f deg el_err=%.3f deg",
                az_err, el_err,
            )

        return pos

    async def get_position(self) -> Position:
        """Return current position from internal step counters."""
        return Position(
            azimuth_deg=self._turntable.position_deg,
            elevation_deg=self._rotor.position_deg,
        )

    async def get_status(self) -> str:
        """Return 'idle' — steppers are synchronous between async calls."""
        return "idle"

    async def trigger_capture(self) -> None:
        """
        No-op stub retained for interface compatibility.

        In GPIO mode the Pi is the capture host; the focus bracket driver is
        called directly by the orchestrator rather than via this method.
        """
        logger.debug("trigger_capture() — no-op in GPIO mode, handled by orchestrator")

    async def home(self) -> Position:
        """Move to (0 deg, 0 deg) and reset internal step counters."""
        logger.info("Homing to (0 deg, 0 deg)")
        pos = await self.move_to(0.0, 0.0)
        self._turntable.reset_position()
        self._rotor.reset_position()
        return pos

    async def execute_scan_plan(
        self,
        positions: list[dict],
        capture_fn=None,
    ) -> list[dict]:
        """
        Execute an ordered list of scan positions.

        Each entry in positions must have 'azimuth' and 'elevation' keys.
        After each move, capture_fn(Position) is awaited if provided.
        Individual position failures are logged but never abort the plan.

        Args:
            positions:  List of dicts with 'azimuth' and 'elevation' keys.
            capture_fn: Async callable invoked after each confirmed move.

        Returns:
            List of result dicts with commanded/confirmed positions, status,
            elapsed time, and any error message.
        """
        results = []
        for i, pos in enumerate(positions):
            az = float(pos["azimuth"])
            el = float(pos["elevation"])
            t_start = time.monotonic()

            logger.info("[%d/%d] az=%.1f el=%.1f", i + 1, len(positions), az, el)
            try:
                confirmed = await self.move_to(az, el)
                if capture_fn:
                    await capture_fn(confirmed)
                elapsed = time.monotonic() - t_start
                results.append({
                    "commanded":  {"azimuth": az, "elevation": el},
                    "confirmed":  {
                        "azimuth_deg":   confirmed.azimuth_deg,
                        "elevation_deg": confirmed.elevation_deg,
                    },
                    "status":    "ok",
                    "elapsed_s": round(elapsed, 2),
                })
                logger.info("  Position %d complete in %.1fs", i + 1, elapsed)

            except Exception as exc:
                elapsed = time.monotonic() - t_start
                logger.error("  Position %d FAILED: %s", i + 1, exc)
                results.append({
                    "commanded": {"azimuth": az, "elevation": el},
                    "confirmed": None,
                    "status":    "failed",
                    "error":     str(exc),
                    "elapsed_s": round(elapsed, 2),
                })

        ok = sum(1 for r in results if r["status"] == "ok")
        logger.info("Scan plan complete: %d/%d succeeded", ok, len(positions))
        return results


# ---------------------------------------------------------------------------
# CLI — acceptance test and manual control
# ---------------------------------------------------------------------------

async def _acceptance_test(dry_run: bool):
    """24-position azimuth sweep at elevation=20 deg (acceptance criterion: +-0.5 deg)."""
    import json

    positions = [{"azimuth": az, "elevation": 20.0} for az in range(0, 360, 15)]

    print(f"\n{'='*60}")
    print("Acceptance Test — 24-position azimuth sweep")
    print(f"Dry-run: {dry_run}")
    print(f"{'='*60}\n")

    async with OpenScanController(dry_run=dry_run) as ctrl:
        t0 = time.monotonic()
        results = await ctrl.execute_scan_plan(positions)
        elapsed = time.monotonic() - t0

    ok     = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    az_errs = [abs(r["confirmed"]["azimuth_deg"]   - r["commanded"]["azimuth"])   for r in ok]
    el_errs = [abs(r["confirmed"]["elevation_deg"] - r["commanded"]["elevation"]) for r in ok]

    max_az = max(az_errs, default=0)
    max_el = max(el_errs, default=0)

    print("Results:")
    print(f"  Positions completed : {len(ok)}/24")
    print(f"  Failures            : {len(failed)}")
    print(f"  Max azimuth error   : {max_az:.4f} deg  {'PASS' if max_az <= 0.5 else 'FAIL'}")
    print(f"  Max elevation error : {max_el:.4f} deg  {'PASS' if max_el <= 0.5 else 'FAIL'}")
    print(f"  Total elapsed       : {elapsed:.1f}s")

    passed = len(failed) == 0 and max_az <= 0.5 and max_el <= 0.5
    print(f"\nAcceptance: {'PASS' if passed else 'FAIL'}")

    out = "acceptance_results.json"
    with open(out, "w") as f:
        json.dump({
            "test":               "gpio_24pos_sweep",
            "dry_run":            dry_run,
            "passed":             passed,
            "elapsed_s":          round(elapsed, 2),
            "max_az_error_deg":   round(max_az, 4),
            "max_el_error_deg":   round(max_el, 4),
            "positions":          results,
        }, f, indent=2)
    print(f"Results written to {out}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="OpenScan RPi GPIO Motor Controller")
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--acceptance-test", action="store_true")
    parser.add_argument("--move", nargs=2, type=float, metavar=("AZ", "EL"))
    parser.add_argument("--home",            action="store_true")
    args = parser.parse_args()

    async def main():
        if args.acceptance_test:
            await _acceptance_test(args.dry_run)
        elif args.move:
            async with OpenScanController(dry_run=args.dry_run) as ctrl:
                pos = await ctrl.move_to(args.move[0], args.move[1])
                print(f"Confirmed: {pos}")
        elif args.home:
            async with OpenScanController(dry_run=args.dry_run) as ctrl:
                pos = await ctrl.home()
                print(f"Homed: {pos}")
        else:
            parser.print_help()

    asyncio.run(main())
