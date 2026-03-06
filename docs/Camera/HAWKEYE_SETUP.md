# Arducam Hawkeye 64MP — Raspberry Pi Setup Notes

Hardware target: Arducam Hawkeye 64MP (OV64A40 sensor), Raspberry Pi 4/5,
Raspberry Pi OS Bookworm 64-bit.

---

## Why this doc exists

The Hawkeye requires Arducam's custom libcamera fork rather than the
standard Pi libcamera stack. The install process has several non-obvious
steps and a persistent gotcha around OS updates. These notes capture what
was learned during initial setup so future sessions don't repeat the same
debugging.

---

## Prerequisites

- Raspberry Pi OS **Bookworm** 64-bit (Buster is not supported; Bookworm
  and Trixie are the recommended targets per Arducam).
- Camera ribbon connected to the CSI port before boot.
- Internet access on the Pi.

---

## Installation

### 1. Disable automatic camera detection

Before installing drivers, edit `/boot/firmware/config.txt` (Bookworm) or
`/boot/config.txt` (older releases):

```
# Disable auto-detect so the Pi doesn't try to probe for an official camera
camera_auto_detect=0

[all]
dtoverlay=arducam-64mp
```

If you are using the **CAM0** port on a Pi 5, use:

```
dtoverlay=arducam-64mp,cam0
```

Reboot after editing.

### 2. Download and run the Arducam install script

Arducam provides a script that installs three components in the correct
order. Do **not** install them manually out of order — the kernel driver
must match the libcamera version.

```bash
wget -O install_pivariety_pkgs.sh \
    https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/latest/download/install_pivariety_pkgs.sh
chmod +x install_pivariety_pkgs.sh

# Install in this exact order:
./install_pivariety_pkgs.sh -p libcamera_dev
./install_pivariety_pkgs.sh -p libcamera_apps
./install_pivariety_pkgs.sh -p arducam_64mp
```

Reboot after the kernel driver install.

### 3. Verify

```bash
libcamera-hello --list-cameras
# Should show: arducam_64mp [...] (9152x6944)

libcamera-still -o test.jpg --width 1920 --height 1080
```

---

## Lessons learned / gotchas

### apt upgrade breaks the driver

**This is the biggest pain point.** Arducam's libcamera fork has not been
upstreamed into the standard Pi libcamera package. Every `apt upgrade` that
bumps `libcamera` or `libpisp` will overwrite Arducam's custom build and
break the camera.

Symptoms after an upgrade:
- `libcamera-hello` reports no cameras found.
- `dmesg` shows `No static properties available for 'arducam_64mp'`.
- picamera2 raises `RuntimeError: Camera ... is not available`.

**Fix:** re-run the install script after any kernel or libcamera upgrade:

```bash
./install_pivariety_pkgs.sh -p libcamera_dev
./install_pivariety_pkgs.sh -p libcamera_apps
./install_pivariety_pkgs.sh -p arducam_64mp
sudo reboot
```

**Mitigation:** pin the Arducam packages to prevent apt from overwriting them:

```bash
sudo apt-mark hold libcamera-dev libcamera-apps
```

Note that this will also block security updates to those packages — weigh
the trade-off for your environment.

### camera_auto_detect must be 0

If `camera_auto_detect=1` is still set (the Pi default), the Pi's own
camera probe runs before the Arducam overlay loads and the camera will
not be detected. This is the most common cause of "no cameras found" on
a fresh install.

### Memory allocation errors at full resolution

Capturing at the full 9152×6944 resolution on a Pi 4 with the desktop
environment running can trigger `Cannot allocate memory` errors in
picamera2 due to DMA buffer exhaustion.

Workarounds in order of preference:

1. **Use Raspberry Pi OS Lite** (headless) — frees the most memory.
2. **Increase GPU memory split** in `config.txt`: `gpu_mem=256`
3. **Capture at reduced resolution** for tasks that don't need full 64MP
   (e.g. focus bracketing quality checks use 1152×868 — 1/8 scale).
4. **Limit `buffer_count`** in the picamera2 configuration:
   ```python
   config = cam.create_still_configuration(..., buffer_count=1)
   ```

### LensPosition scale is 0–15, not 0–1

The Hawkeye's manual focus control uses a `LensPosition` value in the
range **0.0–15.0** (not 0.0–1.0 as documented in some generic picamera2
guides). Direction is inverted relative to physical distance:

| LensPosition | Focus distance |
|---|---|
| 0.0 | Infinity (far) |
| 7.5 | Mid-range (~40–60 cm typical) |
| 15.0 | ~15 cm (closest) |

```python
cam.set_controls({
    "AfMode":       libcamera.controls.AfModeEnum.Manual,
    "LensPosition": 7.5,
})
```

Allow 100 ms settle time after each `LensPosition` change before capturing.

### Exposure lock required for focus bracketing

With AE enabled, exposure changes between focus steps as the depth of
field shifts slightly. Lock exposure after an initial convergence period:

```python
cam.start()
time.sleep(2.0)                        # let AE/AWB converge
meta = cam.capture_metadata()
cam.set_controls({
    "AeEnable":     False,
    "AwbEnable":    False,
    "ExposureTime": meta["ExposureTime"],
    "AnalogueGain": meta["AnalogueGain"],
    "ColourGains":  meta.get("ColourGains", (1.0, 1.0)),
})
time.sleep(0.5)                        # let locked values take effect
```

### Camera is mounted 180° rotated on OpenScan Classic

The Hawkeye is physically mounted upside-down on the OpenScan Classic arm.
Correct at the libcamera level (not in post) so that coordinates in
metadata match the physical orientation:

```python
import libcamera
transform = libcamera.Transform(hflip=1, vflip=1)
config = cam.create_still_configuration(..., transform=transform)
```

### rpicam-still and picamera2 cannot share the camera

`rpicam-still` requires exclusive camera access. If picamera2 holds the
camera open, `rpicam-still` will fail with a device busy error (and vice
versa). For DNG capture via `rpicam-still`, stop picamera2 first:

```python
cam.stop()
subprocess.run(["rpicam-still", "--encoding", "dng", ...], check=True)
cam.start()
time.sleep(0.5)
```

---

## Reference

- [Arducam 64MP Hawkeye official docs](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/64MP-Hawkeye/)
- [Arducam Pivariety V4L2 driver (GitHub)](https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver)
- [Arducam troubleshooting guide](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/Troubleshooting/)
- [Jeff Geerling's setup gist](https://gist.github.com/geerlingguy/de62619a906803808edddaf8bb9cdce8)
- [picamera2 issue #891 — Cannot allocate memory on Bookworm](https://github.com/raspberrypi/picamera2/issues/891)
- [libpisp upgrade breaks Arducam driver (issue #53)](https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/issues/53)
