import json
import pytest
from pathlib import Path

from openscan_firmware.models.pose_metadata import (
    PoseMetadata,
    METADATA_VERSION,
    REQUIRED_FIELDS,
    write_sidecar,
    read_sidecar,
    validate_sidecar,
    sidecar_path_for,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_meta(**overrides) -> PoseMetadata:
    defaults = dict(
        azimuth_deg=45.0,
        elevation_deg=20.0,
        focus_bracket_index=2,
        focus_bracket_total=5,
        sensor_resolution=(9152, 6944),
        session_name="test_session",
        file_path="raw/test_frame.png",
        focus_lens_position=8.5,
        radius_mm=185.0,
        camera_metadata={
            "ExposureTime": 10000,
            "AnalogueGain": 2.0,
            "ColourGains": (1.8, 1.5),
        },
    )
    defaults.update(overrides)
    return PoseMetadata.from_capture(**defaults)


# ---------------------------------------------------------------------------
# PoseMetadata.from_capture
# ---------------------------------------------------------------------------

class TestFromCapture:
    def test_image_id_format(self):
        meta = make_meta(azimuth_deg=45.0, elevation_deg=20.0,
                         focus_bracket_index=2, session_name="scan_001")
        assert meta.image_id == "scan_001_az045.00_el020.00_f2"

    def test_azimuth_rounded_to_4dp(self):
        meta = make_meta(azimuth_deg=45.123456789)
        assert meta.azimuth_deg == pytest.approx(45.1235, abs=1e-6)

    def test_elevation_rounded_to_4dp(self):
        meta = make_meta(elevation_deg=20.987654321)
        assert meta.elevation_deg == pytest.approx(20.9877, abs=1e-6)

    def test_camera_metadata_extracted(self):
        meta = make_meta(camera_metadata={
            "ExposureTime": 12345,
            "AnalogueGain": 3.0,
            "ColourGains": (2.0, 1.2),
        })
        assert meta.exposure_time_us == 12345
        assert meta.analogue_gain    == pytest.approx(3.0)
        assert meta.colour_gains     == pytest.approx((2.0, 1.2))

    def test_none_camera_metadata(self):
        meta = make_meta(camera_metadata=None)
        assert meta.exposure_time_us is None
        assert meta.analogue_gain    is None
        assert meta.colour_gains     is None

    def test_version_set(self):
        meta = make_meta()
        assert meta.version == METADATA_VERSION

    def test_timestamp_utc_set(self):
        meta = make_meta()
        assert meta.timestamp_utc is not None
        assert "T" in meta.timestamp_utc  # ISO-8601

    def test_optional_radius_mm(self):
        meta = make_meta(radius_mm=200.0)
        assert meta.radius_mm == pytest.approx(200.0)

    def test_optional_radius_mm_none(self):
        meta = make_meta(radius_mm=None)
        assert meta.radius_mm is None


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_returns_dict(self):
        assert isinstance(make_meta().to_dict(), dict)

    def test_sensor_resolution_is_list(self):
        d = make_meta().to_dict()
        assert isinstance(d["sensor_resolution"], list)

    def test_colour_gains_is_list(self):
        d = make_meta().to_dict()
        assert isinstance(d["colour_gains"], list)

    def test_required_fields_present(self):
        d = make_meta().to_dict()
        for field in REQUIRED_FIELDS:
            assert field in d, f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# write_sidecar / read_sidecar round-trip
# ---------------------------------------------------------------------------

class TestSidecarRoundTrip:
    def test_write_creates_file(self, tmp_path):
        meta = make_meta()
        path = tmp_path / "frame.json"
        write_sidecar(meta, path)
        assert path.exists()

    def test_write_is_valid_json(self, tmp_path):
        meta = make_meta()
        path = tmp_path / "frame.json"
        write_sidecar(meta, path)
        json.loads(path.read_text())  # should not raise

    def test_roundtrip_azimuth(self, tmp_path):
        meta = make_meta(azimuth_deg=123.456)
        path = tmp_path / "frame.json"
        write_sidecar(meta, path)
        meta2 = read_sidecar(path)
        assert meta2.azimuth_deg == pytest.approx(meta.azimuth_deg, abs=1e-6)

    def test_roundtrip_elevation(self, tmp_path):
        meta = make_meta(elevation_deg=55.5)
        path = tmp_path / "frame.json"
        write_sidecar(meta, path)
        meta2 = read_sidecar(path)
        assert meta2.elevation_deg == pytest.approx(meta.elevation_deg, abs=1e-6)

    def test_roundtrip_image_id(self, tmp_path):
        meta = make_meta()
        path = tmp_path / "frame.json"
        write_sidecar(meta, path)
        meta2 = read_sidecar(path)
        assert meta2.image_id == meta.image_id

    def test_roundtrip_sensor_resolution_is_tuple(self, tmp_path):
        meta = make_meta()
        path = tmp_path / "frame.json"
        write_sidecar(meta, path)
        meta2 = read_sidecar(path)
        assert isinstance(meta2.sensor_resolution, tuple)
        assert meta2.sensor_resolution == (9152, 6944)

    def test_roundtrip_colour_gains_is_tuple(self, tmp_path):
        meta = make_meta()
        path = tmp_path / "frame.json"
        write_sidecar(meta, path)
        meta2 = read_sidecar(path)
        assert isinstance(meta2.colour_gains, tuple)

    def test_atomic_write_no_partial_file(self, tmp_path):
        """Temp file should not exist after a successful write."""
        meta = make_meta()
        path = tmp_path / "frame.json"
        write_sidecar(meta, path)
        tmp = path.with_suffix(".json.tmp")
        assert not tmp.exists()

    def test_read_ignores_unknown_fields(self, tmp_path):
        """Extra fields in a sidecar should not raise on read."""
        meta = make_meta()
        path = tmp_path / "frame.json"
        write_sidecar(meta, path)
        data = json.loads(path.read_text())
        data["future_field"] = "some_value"
        path.write_text(json.dumps(data))
        read_sidecar(path)  # should not raise


# ---------------------------------------------------------------------------
# validate_sidecar
# ---------------------------------------------------------------------------

class TestValidateSidecar:
    def test_valid_sidecar_passes(self, tmp_path):
        path = tmp_path / "frame.json"
        write_sidecar(make_meta(), path)
        assert validate_sidecar(path) is True

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            validate_sidecar(tmp_path / "nonexistent.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{ not valid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            validate_sidecar(path)

    def test_missing_required_field_raises(self, tmp_path):
        path = tmp_path / "frame.json"
        write_sidecar(make_meta(), path)
        data = json.loads(path.read_text())
        del data["azimuth_deg"]
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="missing fields"):
            validate_sidecar(path)

    def test_wrong_version_raises(self, tmp_path):
        path = tmp_path / "frame.json"
        write_sidecar(make_meta(), path)
        data = json.loads(path.read_text())
        data["version"] = "99.0"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="version"):
            validate_sidecar(path)

    def test_invalid_azimuth_raises(self, tmp_path):
        path = tmp_path / "frame.json"
        write_sidecar(make_meta(), path)
        data = json.loads(path.read_text())
        data["azimuth_deg"] = 999.0
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="azimuth"):
            validate_sidecar(path)

    def test_invalid_elevation_raises(self, tmp_path):
        path = tmp_path / "frame.json"
        write_sidecar(make_meta(), path)
        data = json.loads(path.read_text())
        data["elevation_deg"] = 200.0
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="elevation"):
            validate_sidecar(path)

    def test_invalid_sensor_resolution_raises(self, tmp_path):
        path = tmp_path / "frame.json"
        write_sidecar(make_meta(), path)
        data = json.loads(path.read_text())
        data["sensor_resolution"] = [9152]  # only one element
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="sensor_resolution"):
            validate_sidecar(path)


# ---------------------------------------------------------------------------
# sidecar_path_for
# ---------------------------------------------------------------------------

class TestSidecarPathFor:
    def test_replaces_suffix_with_json(self):
        p = Path("raw/scan_az045.00_el020.00_f2.png")
        assert sidecar_path_for(p) == Path("raw/scan_az045.00_el020.00_f2.json")

    def test_works_with_tiff(self):
        p = Path("raw/frame.tiff")
        assert sidecar_path_for(p).suffix == ".json"
