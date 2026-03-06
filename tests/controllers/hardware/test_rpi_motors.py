import asyncio
import pytest
from unittest.mock import MagicMock, patch, call

from openscan_firmware.controllers.hardware.rpi_motors import (
    MotorConfig,
    StepperMotor,
    Position,
    OpenScanController,
    TURNTABLE_CFG,
    ROTOR_CFG,
    SETTLE_DELAY_S,
)

# ---------------------------------------------------------------------------
# Patch paths
# ---------------------------------------------------------------------------

TIME_PATCH  = "openscan_firmware.controllers.hardware.rpi_motors.time"
ASYNCIO_PATCH = "openscan_firmware.controllers.hardware.rpi_motors.asyncio"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def motor_cfg():
    return MotorConfig(
        step_pin=11, dir_pin=9, en_pin=22,
        steps_per_rot=3200, accel_ramp=50,
        step_delay=0.0001, direction=1,
    )


@pytest.fixture
def mock_gpio():
    gpio = MagicMock()
    gpio.OUT   = 1
    gpio.LOW   = 0
    gpio.HIGH  = 1
    return gpio


@pytest.fixture
def stepper(motor_cfg, mock_gpio):
    return StepperMotor(cfg=motor_cfg, gpio=mock_gpio, dry_run=False)


@pytest.fixture
def stepper_dry(motor_cfg):
    return StepperMotor(cfg=motor_cfg, gpio=None, dry_run=True)


# ---------------------------------------------------------------------------
# StepperMotor — unit tests
# ---------------------------------------------------------------------------

class TestStepperMotorConversions:
    def test_deg_to_steps_zero(self, stepper):
        assert stepper._deg_to_steps(0.0) == 0

    def test_deg_to_steps_full_rotation(self, stepper):
        assert stepper._deg_to_steps(360.0) == 3200

    def test_deg_to_steps_quarter(self, stepper):
        assert stepper._deg_to_steps(90.0) == 800

    def test_steps_to_deg_full(self, stepper):
        assert stepper._steps_to_deg(3200) == pytest.approx(360.0)

    def test_steps_to_deg_quarter(self, stepper):
        assert stepper._steps_to_deg(800) == pytest.approx(90.0)

    def test_roundtrip(self, stepper):
        for deg in [0.0, 15.0, 45.0, 90.0, 180.0, 270.0, 359.0]:
            steps = stepper._deg_to_steps(deg)
            assert stepper._steps_to_deg(steps) == pytest.approx(deg, abs=0.2)


class TestStepperMotorRamp:
    def test_ramp_at_start_is_slow(self, stepper):
        """First step should use the 4x multiplier."""
        delay = stepper._step_delay_for(0, 1000)
        assert delay == pytest.approx(stepper.cfg.step_delay * 4.0)

    def test_ramp_at_end_is_slow(self, stepper):
        """Last step should use a significantly elevated delay (ramp deceleration)."""
        delay = stepper._step_delay_for(999, 1000)
        assert delay >= stepper.cfg.step_delay * 3.5

    def test_ramp_at_midpoint_is_fast(self, stepper):
        """Middle of a long move should be at or near 1x base delay."""
        delay = stepper._step_delay_for(500, 1000)
        assert delay <= stepper.cfg.step_delay * 1.1

    def test_ramp_delay_is_positive(self, stepper):
        for i in [0, 1, 25, 50, 99]:
            assert stepper._step_delay_for(i, 100) > 0


class TestStepperMotorGpioInit:
    def test_gpio_pins_configured_on_init(self, motor_cfg, mock_gpio):
        StepperMotor(cfg=motor_cfg, gpio=mock_gpio, dry_run=False)
        mock_gpio.setup.assert_any_call(motor_cfg.step_pin, mock_gpio.OUT, initial=mock_gpio.LOW)
        mock_gpio.setup.assert_any_call(motor_cfg.dir_pin,  mock_gpio.OUT, initial=mock_gpio.LOW)
        mock_gpio.setup.assert_any_call(motor_cfg.en_pin,   mock_gpio.OUT, initial=mock_gpio.HIGH)

    def test_dry_run_skips_gpio_init(self, motor_cfg):
        mock_gpio = MagicMock()
        StepperMotor(cfg=motor_cfg, gpio=mock_gpio, dry_run=True)
        mock_gpio.setup.assert_not_called()


class TestStepperMotorPosition:
    def test_initial_position_is_zero(self, stepper):
        assert stepper.position_deg == pytest.approx(0.0)

    def test_reset_position(self, stepper):
        stepper._position_steps = 800
        stepper.reset_position()
        assert stepper._position_steps == 0
        assert stepper.position_deg == pytest.approx(0.0)


@pytest.mark.asyncio
class TestStepperMotorDryRun:
    async def test_move_to_deg_updates_position(self, stepper_dry):
        await stepper_dry.move_to_deg(90.0)
        assert stepper_dry.position_deg == pytest.approx(90.0, abs=0.2)

    async def test_move_to_deg_no_op_when_already_at_target(self, stepper_dry):
        result = await stepper_dry.move_to_deg(0.0)
        assert result == pytest.approx(0.0)

    async def test_multiple_moves_accumulate(self, stepper_dry):
        await stepper_dry.move_to_deg(90.0)
        await stepper_dry.move_to_deg(180.0)
        assert stepper_dry.position_deg == pytest.approx(180.0, abs=0.2)

    async def test_returns_confirmed_position(self, stepper_dry):
        result = await stepper_dry.move_to_deg(45.0)
        assert result == pytest.approx(45.0, abs=0.2)


@pytest.mark.asyncio
class TestStepperMotorLiveGpio:
    async def test_blocking_move_enables_then_disables(self, stepper, mock_gpio):
        with patch(TIME_PATCH + ".sleep"):
            await stepper.move_to_deg(90.0)

        # EN pin should have been set LOW (enable) then HIGH (disable)
        en = stepper.cfg.en_pin
        calls = mock_gpio.output.call_args_list
        en_calls = [c for c in calls if c.args[0] == en]
        assert any(c.args[1] == mock_gpio.LOW  for c in en_calls), "EN never enabled"
        assert any(c.args[1] == mock_gpio.HIGH for c in en_calls), "EN never disabled"

    async def test_blocking_move_pulses_step_pin(self, stepper, mock_gpio):
        with patch(TIME_PATCH + ".sleep"):
            await stepper.move_to_deg(90.0)  # 800 steps

        step = stepper.cfg.step_pin
        highs = [c for c in mock_gpio.output.call_args_list
                 if c.args[0] == step and c.args[1] == mock_gpio.HIGH]
        assert len(highs) == 800

    async def test_direction_pin_set_for_positive_move(self, stepper, mock_gpio):
        with patch(TIME_PATCH + ".sleep"):
            await stepper.move_to_deg(90.0)

        dir_calls = [c for c in mock_gpio.output.call_args_list
                     if c.args[0] == stepper.cfg.dir_pin]
        assert dir_calls[0].args[1] == mock_gpio.HIGH

    async def test_direction_pin_set_for_negative_move(self, stepper, mock_gpio):
        stepper._position_steps = 800  # start at 90 deg
        with patch(TIME_PATCH + ".sleep"):
            await stepper.move_to_deg(0.0)

        dir_calls = [c for c in mock_gpio.output.call_args_list
                     if c.args[0] == stepper.cfg.dir_pin]
        assert dir_calls[0].args[1] == mock_gpio.LOW


# ---------------------------------------------------------------------------
# OpenScanController — unit tests (dry-run)
# ---------------------------------------------------------------------------

@pytest.fixture
def ctrl_dry():
    return OpenScanController(dry_run=True)


@pytest.mark.asyncio
class TestOpenScanControllerDryRun:
    async def test_context_manager_enters_and_exits(self, ctrl_dry):
        async with ctrl_dry as ctrl:
            assert ctrl._turntable is not None
            assert ctrl._rotor is not None

    async def test_get_status_returns_idle(self, ctrl_dry):
        async with ctrl_dry as ctrl:
            assert await ctrl.get_status() == "idle"

    async def test_move_to_returns_position(self, ctrl_dry):
        async with ctrl_dry as ctrl:
            pos = await ctrl.move_to(45.0, 20.0)
        assert isinstance(pos, Position)
        assert pos.azimuth_deg   == pytest.approx(45.0, abs=0.2)
        assert pos.elevation_deg == pytest.approx(20.0, abs=0.2)

    async def test_get_position_after_move(self, ctrl_dry):
        async with ctrl_dry as ctrl:
            await ctrl.move_to(90.0, 30.0)
            pos = await ctrl.get_position()
        assert pos.azimuth_deg   == pytest.approx(90.0, abs=0.2)
        assert pos.elevation_deg == pytest.approx(30.0, abs=0.2)

    async def test_home_resets_to_zero(self, ctrl_dry):
        async with ctrl_dry as ctrl:
            await ctrl.move_to(90.0, 30.0)
            pos = await ctrl.home()
        assert pos.azimuth_deg   == pytest.approx(0.0, abs=0.2)
        assert pos.elevation_deg == pytest.approx(0.0, abs=0.2)

    async def test_trigger_capture_is_noop(self, ctrl_dry):
        async with ctrl_dry as ctrl:
            await ctrl.trigger_capture()  # should not raise

    async def test_execute_scan_plan_all_succeed(self, ctrl_dry):
        positions = [{"azimuth": az, "elevation": 20.0} for az in range(0, 90, 30)]
        async with ctrl_dry as ctrl:
            results = await ctrl.execute_scan_plan(positions)
        assert len(results) == 3
        assert all(r["status"] == "ok" for r in results)

    async def test_execute_scan_plan_calls_capture_fn(self, ctrl_dry):
        captured = []

        async def capture_fn(pos):
            captured.append(pos)

        positions = [{"azimuth": 0.0, "elevation": 0.0},
                     {"azimuth": 45.0, "elevation": 10.0}]
        async with ctrl_dry as ctrl:
            await ctrl.execute_scan_plan(positions, capture_fn=capture_fn)

        assert len(captured) == 2

    async def test_execute_scan_plan_continues_on_failure(self, ctrl_dry):
        call_count = 0

        async def failing_fn(pos):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated failure")

        positions = [{"azimuth": 0.0, "elevation": 0.0},
                     {"azimuth": 45.0, "elevation": 0.0}]
        async with ctrl_dry as ctrl:
            results = await ctrl.execute_scan_plan(positions, capture_fn=failing_fn)

        assert results[0]["status"] == "failed"
        assert results[1]["status"] == "ok"
        assert "error" in results[0]

    async def test_execute_scan_plan_result_fields(self, ctrl_dry):
        positions = [{"azimuth": 15.0, "elevation": 5.0}]
        async with ctrl_dry as ctrl:
            results = await ctrl.execute_scan_plan(positions)

        r = results[0]
        assert "commanded"  in r
        assert "confirmed"  in r
        assert "status"     in r
        assert "elapsed_s"  in r
        assert r["commanded"]["azimuth"]   == 15.0
        assert r["commanded"]["elevation"] == 5.0


class TestOpenScanControllerRpigpioImport:
    def test_missing_rpigpio_raises_helpful_error(self):
        """Without RPi.GPIO available, entering the context should raise RuntimeError."""
        ctrl = OpenScanController(dry_run=False)
        with patch.dict("sys.modules", {"RPi": None, "RPi.GPIO": None}):
            with pytest.raises((RuntimeError, ImportError)):
                asyncio.run(ctrl.__aenter__())


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------

class TestPosition:
    def test_str_representation(self):
        pos = Position(azimuth_deg=45.0, elevation_deg=20.0)
        s = str(pos)
        assert "45" in s
        assert "20" in s

    def test_fields(self):
        pos = Position(azimuth_deg=90.0, elevation_deg=0.0)
        assert pos.azimuth_deg   == 90.0
        assert pos.elevation_deg == 0.0


# ---------------------------------------------------------------------------
# Default config constants
# ---------------------------------------------------------------------------

class TestDefaultConfigs:
    def test_turntable_steps_per_rot(self):
        assert TURNTABLE_CFG.steps_per_rot == 3200

    def test_rotor_steps_per_rot(self):
        assert ROTOR_CFG.steps_per_rot == 17067

    def test_settle_delay_is_positive(self):
        assert SETTLE_DELAY_S > 0
