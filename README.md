# Litter Event Logger — local run

## 1. Set up a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Point it at your files

Either edit the `DEFAULT_*` constants near the top of `litter_event_logger.py`
(so you can just hit Run in Antigravity), **or** pass CLI args:

```bash
python litter_event_logger.py \
    --weights "weights/best_int8.onnx" \
    --video "test vid/trash stock.webm" \
    --output "runs/EventLogger"
```

Useful flags:
- `--headless` — run with no GUI window (recommended for background service / boot mode)
- `--camera-id 10` — video capture device index (default: `10` for `/dev/video10` v4l2loopback shared camera; set `0` for direct physical webcam)
- `--distance-interval 10.0` — geodesic distance in meters between captures (default: 10.0m; set 0 for continuous mode)
- `--gnss-timeout 15.0` — grace period in seconds on GNSS loss before falling back to clock (default: 15.0s)
- `--fallback-interval 30.0` — time interval in seconds between captures during clock fallback (default: 30.0s)
- `--trigger-mode {auto,distance,clock,continuous}` — operational mode (default: `auto`)
- `--stride N` — run YOLO detection every Nth frame (e.g. `--stride 2` cuts Pi CPU load)
- `--aod X1 Y1 X2 Y2` — Area of Disinterest box (default: `500 1000 1200 3500`)
- `--overlap-threshold 0.5` — ignore detections overlapping the AoD by more than this
- `--conf 0.25` — YOLO confidence threshold
- `--latitude` / `--longitude` — optional manual coordinates override (by default gathered via GNSS/NavCast)
- `--show` — pop up a live preview window while it processes (press `q` to quit, `space`/`c` for manual trigger)

## 3. Run

```bash
# GUI mode (reads from /dev/video10 shared camera with 10m GNSS distance-based trigger + preview)
python litter_event_logger.py --show

# Headless mode (reads from /dev/video10 shared camera, runs on vehicle boot as background service)
python litter_event_logger.py --headless

# Direct physical webcam test (bypasses loopback relay; uses /dev/video0)
python litter_event_logger.py --camera-id 0 --show

# Continuous video processing mode (process every frame of video)
python litter_event_logger.py --no-webcam --video "test vid/trash stock.webm" --trigger-mode continuous --show
```

## 4. Run on Boot as a Systemd Service (Raspberry Pi / Linux)

To enable autonomous startup on vehicle boot:
```bash
# 1. Install and enable the systemd service (auto-starts on boot)
sudo bash install_service.sh
```

`install_service.sh` automatically configures:
- `ExecStart` passing `--camera-id 10` for `/dev/video10`.
- Systemd dependencies: `Requires=camera-relay.service` and `After=network-online.target time-sync.target camera-relay.service` so `litter-detection` starts after the camera relay is active.

### Controlling the Service via Linux Terminal

Use standard `systemctl` commands to manage the background service:

```bash
# Start the service
sudo systemctl start litter-detection

# Stop the service
sudo systemctl stop litter-detection

# Restart / Reboot the service
sudo systemctl restart litter-detection

# Check live service status & health
sudo systemctl status litter-detection

# View live real-time output and detection logs
sudo journalctl -u litter-detection -f

# Completely uninstall and disable the service
sudo bash uninstall_service.sh
```

## 5. Dual-App Shared Camera Setup (v4l2loopback `/dev/video10`)

The physical Logitech C920 USB webcam can only be opened by a single process at a time under Linux V4L2. Since both **Litter-Detection** and **MotionFull** need to consume the video stream concurrently on the vehicle, a virtual V4L2 loopback device (`/dev/video10`) is utilized:

1. **Camera Relay**: `camera-relay.service` uses `ffmpeg` to read frames from the physical webcam (`/dev/v4l/by-id/...`) and continuously pipes them into `/dev/video10`.
2. **Concurrent Consumers**: Both `litter-detection` and `MotionFull` attach as independent consumers to `/dev/video10` without locking conflicts or frame starvation.
3. **Pre-Configured**: `litter_event_logger.py` defaults to `DEFAULT_CAMERA_ID = 10`, and `install_service.sh` hooks into `camera-relay.service`.

For full step-by-step setup of the `v4l2loopback` kernel module, modprobe config, and `camera-relay.service`, refer to [v412loopback setup.txt](file:///c:/Users/Āḍṁīṇ/Desktop/Solid-Waste/Litter-Detection-main/Litter-Detection-main/v412loopback%20setup.txt).

## 6. Orange Status LED (Raspberry Pi GPIO 25 / Physical Pin 22)

Connect a single Orange LED with a 220–330 Ohm resistor:
- **Anode (+)**: Physical Pin 22 (BCM GPIO 25)
- **Cathode (-)**: Physical Pin 20 (GND)

| LED Pattern | Meaning |
|---|---|
| **Solid Light** | System functional (ready, 3D GNSS locked, 10m distance tracking active) |
| **Constant Blinking** (1 Hz) | Scanning for GNSS (boot scan or mid-run loss grace period) |
| **Triple Blink** (3 fast pulses) | Capturing a frame on trigger |
| **Heartbeat Blink** (pulse-pulse-pause) | Error faced (camera fail, hardware fault, exception) |

Annotated frames and machine-readable sidecar JSON files land in `runs/EventLogger/`.
The image banner matches the reference format (Yellow `RTC: ...` and Green `GNSS: ...`, with zero developer name, appended below the image so 100% of frame pixels are preserved).



## Notes on what changed from the notebook

- Removed the Colab-only `!pip install ultralytics` cell — installs now happen once via `requirements.txt`.
- Removed hardcoded `/content/drive/MyDrive/...` paths — replaced with CLI args / editable defaults.
- Added basic file-existence checks so it fails with a clear message instead of an OpenCV error if a path is wrong.
- Added `Ctrl+C` / `q`-key handling so the video capture always releases cleanly.
- Logic (overlap check, ByteTrack, per-track dedup, frame annotation) is unchanged.
