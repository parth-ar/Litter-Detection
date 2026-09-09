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
# GUI mode (10m GNSS distance-based trigger + interactive viewfinder)
python litter_event_logger.py --show

# Headless mode (runs on vehicle boot as background service)
python litter_event_logger.py --headless

# Continuous video processing mode (process every frame of video)
python litter_event_logger.py --no-webcam --video "test vid/trash stock.webm" --trigger-mode continuous --show
```

## 4. Run on Boot as a Systemd Service (Raspberry Pi / Linux)

To enable autonomous startup on vehicle boot:
```bash
# 1. Install and enable the systemd service (auto-starts on boot)
sudo bash install_service.sh
```

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


## 5. Orange Status LED (Raspberry Pi GPIO 25 / Physical Pin 22)

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
