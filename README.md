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
- `--headless` — run with no GUI window (perfect for running alongside other OpenCV apps or background services)
- `--stride N` — run YOLO detection every Nth frame (e.g. `--stride 2` cuts Pi CPU load by 50%)
- `--aod X1 Y1 X2 Y2` — Area of Disinterest box (default matches your notebook: `500 1000 1200 3500`)
- `--overlap-threshold 0.5` — ignore detections overlapping the AoD by more than this
- `--conf 0.25` — YOLO confidence threshold
- `--latitude` / `--longitude` — optional manual coordinates override (by default coordinates are gathered live via GNSS/NavCast)
- `--show` — pop up a live preview window while it processes (press `q` to stop early)

## 3. Run

```bash
# GUI mode (interactive AoD polygon setup + live viewfinder)
python litter_event_logger.py --show

# Headless mode (concurrent OpenCV programs / vehicle background service)
python litter_event_logger.py --headless --stride 2
```

Annotated frames land in the output folder as `Frame_<n>.jpg`, same as before.

## Notes on what changed from the notebook

- Removed the Colab-only `!pip install ultralytics` cell — installs now happen once via `requirements.txt`.
- Removed hardcoded `/content/drive/MyDrive/...` paths — replaced with CLI args / editable defaults.
- Added basic file-existence checks so it fails with a clear message instead of an OpenCV error if a path is wrong.
- Added `Ctrl+C` / `q`-key handling so the video capture always releases cleanly.
- Logic (overlap check, ByteTrack, per-track dedup, frame annotation) is unchanged.
