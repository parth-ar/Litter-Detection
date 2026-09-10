"""
Litter Detection Event Logger
------------------------------
Runs a YOLO model with ByteTrack on a video or live webcam feed, ignores
detections inside a polygon Area of Disinterest (AoD), and saves one
annotated frame per newly-seen tracked object.

Integrates:
- NavCast GNSS TCP NMEA driver (with automatic phone gateway/port discovery)
- DS3231 RTC module (with kernel / SMBus / software fallback)
- Non-destructive metadata bottom banner (prevents concealing detection pixels)
- Telemetry sidecar JSON per saved event

PHASES (webcam mode with --draw-aod / DEFAULT_DRAW_AOD = True)
--------------------------------------------------------------
1. SETUP  -- live camera feed shown in the window.
             Draw your AoD polygon by clicking.
             Press Enter/Space to confirm and start detection.
             Press Esc to skip (no AoD will be used).

2. DETECT -- YOLO + ByteTrack runs in real-time.
             Detections inside the polygon are greyed-out / ignored.
             Keyboard:  q = quit   t = pause/resume

Usage - no args (uses webcam + polygon drawing by default):
    python litter_event_logger.py

Usage - video file:
    python litter_event_logger.py --no-webcam --show

Usage - webcam, skip polygon drawing:
    python litter_event_logger.py --no-draw-aod
"""

import argparse
import json
import os
import sys
import time
import queue
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

# Prefer lightweight, zero-PyTorch detector on Python 3.14 / Pi; fall back to ultralytics
try:
    from detector import YOLO
except ImportError:
    from ultralytics import YOLO

# Ensure repository root is on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config
import sensors.rtc as rtc_sensor
import sensors.gnss as gnss_sensor
import sensors.leds as leds_sensor
from sensors.rtc_sync import periodic_sync_loop


# ---------------------------------------------------------------------------
# Asynchronous Background Disk Writer (Zero save latency impact)
# ---------------------------------------------------------------------------
_save_queue = queue.Queue()

def _save_worker():
    """Background worker that writes JPEG frames and JSON sidecars to disk."""
    while True:
        item = _save_queue.get()
        if item is None:
            _save_queue.task_done()
            break
        filename, final_frame, sidecar_filename, sidecar_data = item
        try:
            cv2.imwrite(filename, final_frame)
            print(f"[EVENT] Saved {filename} (non-concealing banner, {final_frame.shape[1]}x{final_frame.shape[0]})")
            with open(sidecar_filename, "w", encoding="utf-8") as sf:
                json.dump(sidecar_data, sf, indent=2)
        except Exception as e:
            print(f"[EVENT] Error writing {filename}: {e}")
        finally:
            _save_queue.task_done()

# ---------------------------------------------------------------------------
# Defaults  (edit here to run without CLI args in Antigravity)
# ---------------------------------------------------------------------------
_WEIGHT_CANDIDATES = [
    os.path.join("weights", "best_int8.onnx"),
    os.path.join("weights", "best.onnx"),
    os.path.join("weights", "best.pt"),
]
DEFAULT_WEIGHTS = next((p for p in _WEIGHT_CANDIDATES if os.path.isfile(p)), _WEIGHT_CANDIDATES[-1])

DEFAULT_VIDEO     = r"test vid/trash stock.webm"
DEFAULT_OUTPUT    = r"runs/EventLogger"
DEFAULT_CAMERA_ID = 10
DEFAULT_WEBCAM    = True   # False -> use video file

DEFAULT_DRAW_AOD  = True   # False -> skip polygon setup, use rect AoD

DEFAULT_AOD               = (500, 1000, 1200, 3500)  # x1 y1 x2 y2 rect fallback
DEFAULT_OVERLAP_THRESHOLD = 0.50
DEFAULT_TIMEZONE          = config.load_timezone()
DEFAULT_CONF              = 0.25

WINDOW = "Litter Event Logger"



# ---------------------------------------------------------------------------
# AoD geometry helpers
# ---------------------------------------------------------------------------

def polygon_overlap(box, poly_pts, frame_shape):
    """Fraction of the bounding-box that lies inside the polygon AoD."""
    if poly_pts is None or len(poly_pts) < 3:
        return 0.0
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(w, x2), min(h, y2)
    if bx2 <= bx1 or by2 <= by1:
        return 0.0
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly_pts, dtype=np.int32)], 255)
    roi      = mask[by1:by2, bx1:bx2]
    box_area = (bx2 - bx1) * (by2 - by1)
    return float(np.count_nonzero(roi)) / box_area


def rect_overlap(box, aod):
    """Fraction of the bounding-box that overlaps the rectangular AoD."""
    bx1, by1, bx2, by2 = box
    ax1, ay1, ax2, ay2 = aod
    ix1, iy1 = max(bx1, ax1), max(by1, ay1)
    ix2, iy2 = min(bx2, ax2), min(by2, ay2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    box_area = (bx2 - bx1) * (by2 - by1)
    return (ix2 - ix1) * (iy2 - iy1) / box_area if box_area > 0 else 0.0


# ---------------------------------------------------------------------------
# AoD overlay drawing (used during detection phase)
# ---------------------------------------------------------------------------

def draw_polygon_aod(frame, poly_pts):
    """Semi-transparent red polygon overlay in-place."""
    if poly_pts is None or len(poly_pts) < 3:
        return
    pts = np.array(poly_pts, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 0, 180))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
    cx = int(pts[:, 0].mean())
    cy = int(pts[:, 1].mean())
    cv2.putText(frame, "AoD (ignored)", (cx - 60, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


def draw_rect_aod(frame, aod):
    """Semi-transparent red rectangle overlay in-place."""
    ax1, ay1, ax2, ay2 = aod
    h, w = frame.shape[:2]
    ax1, ax2 = max(0, ax1), min(w, ax2)
    ay1, ay2 = max(0, ay1), min(h, ay2)
    if ax2 <= ax1 or ay2 <= ay1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (ax1, ay1), (ax2, ay2), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 0, 255), 2)
    cv2.putText(frame, "AoD (ignored)", (ax1 + 6, ay1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


# ---------------------------------------------------------------------------
# PHASE 1 — Setup: live-feed polygon drawing
# ---------------------------------------------------------------------------

def run_setup_phase(cap):
    """
    Show a live camera feed and let the user draw an AoD polygon.

    Controls
    --------
    Left-click        add vertex
    Right-click       remove last vertex
    Enter / Space     confirm (min 3 points required)
    Esc               skip — no polygon AoD

    Returns list of (x,y) tuples, or [] if skipped.
    """
    points    = []
    mouse_pos = [0, 0]

    def mouse_cb(event, x, y, _flags, _param):
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse_cb)

    warn_until = 0.0

    print("SETUP PHASE: Draw your AoD polygon in the window.")
    print("  Left-click to add points, Right-click to undo.")
    print("  Press Enter/Space to confirm, Esc to skip.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()

        # ---- Draw polygon in progress ----
        if len(points) >= 3:
            overlay = display.copy()
            cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], (0, 0, 160))
            cv2.addWeighted(overlay, 0.30, display, 0.70, 0, display)

        if len(points) >= 2:
            cv2.polylines(display, [np.array(points, dtype=np.int32)],
                          False, (0, 255, 255), 2)

        # Closing preview line (last point -> mouse)
        if points:
            cv2.line(display, points[-1], tuple(mouse_pos), (0, 200, 200), 1)
        # Closing edge preview (last -> first), when >= 2 pts
        if len(points) >= 2:
            cv2.line(display, points[-1], points[0], (0, 200, 200), 1)

        # Vertex dots + labels
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 6, (0, 255, 255), -1)
            cv2.circle(display, pt, 6, (0, 0, 0), 1)
            cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # ---- Top banner ----
        banner_h = 42
        cv2.rectangle(display, (0, 0), (display.shape[1], banner_h), (20, 20, 20), -1)
        cv2.putText(display, "SETUP  -  Draw Area of Disinterest Polygon",
                    (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 220, 255), 1)

        # ---- Bottom instructions ----
        h = display.shape[0]
        cv2.rectangle(display, (0, h - 52), (display.shape[1], h), (20, 20, 20), -1)
        tips = ("Left-click: add point  |  Right-click: undo  |  "
                "Enter / Space: start detection  |  Esc: skip AoD")
        cv2.putText(display, tips, (8, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        cv2.putText(display, f"Points placed: {len(points)}  (need >= 3 to confirm)",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # "Need 3+ points" flash
        import time
        if time.time() < warn_until:
            cv2.putText(display, "Need at least 3 points!",
                        (display.shape[1] // 2 - 140, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (13, 32):          # Enter or Space
            if len(points) >= 3:
                break
            import time
            warn_until = time.time() + 1.5
        elif key == 27:              # Esc - skip AoD
            points = []
            print("AoD skipped. Running without an Area of Disinterest.")
            break

    # Detach mouse callback before switching to detection
    cv2.setMouseCallback(WINDOW, lambda *a: None)
    return points


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Litter detection event/frame logger with NavCast & RTC")
    p.add_argument("--weights",            default=DEFAULT_WEIGHTS)
    p.add_argument("--video",              default=DEFAULT_VIDEO)
    p.add_argument("--output",             default=DEFAULT_OUTPUT)

    # Webcam / video toggle
    src = p.add_mutually_exclusive_group()
    src.add_argument("--webcam",    dest="webcam", action="store_true",
                     default=DEFAULT_WEBCAM, help="Use live webcam (default)")
    src.add_argument("--no-webcam", dest="webcam", action="store_false",
                     help="Use video file instead of webcam")

    p.add_argument("--camera-id",          type=int, default=DEFAULT_CAMERA_ID)

    # AoD draw toggle
    aod = p.add_mutually_exclusive_group()
    aod.add_argument("--draw-aod",    dest="draw_aod", action="store_true",
                     default=DEFAULT_DRAW_AOD,
                     help="Show live setup phase to draw polygon AoD (default)")
    aod.add_argument("--no-draw-aod", dest="draw_aod", action="store_false",
                     help="Skip setup phase, use rectangular AoD")

    p.add_argument("--aod",                nargs=4, type=int,
                    default=list(DEFAULT_AOD),
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="Rectangular AoD fallback")
    p.add_argument("--overlap-threshold",  type=float,
                    default=DEFAULT_OVERLAP_THRESHOLD)
    p.add_argument("--conf",               type=float, default=DEFAULT_CONF)
    p.add_argument("--latitude",           type=float, default=None,
                   help="Manual latitude override (default: None; gathered via GNSS / NavCast)")
    p.add_argument("--longitude",          type=float, default=None,
                   help="Manual longitude override (default: None; gathered via GNSS / NavCast)")
    p.add_argument("--timezone",           default=DEFAULT_TIMEZONE)
    p.add_argument("--show",               action="store_true",
                    help="Show preview window (auto-on for webcam/draw-aod)")
    p.add_argument("--headless",           action="store_true", default=False,
                    help="Run without GUI window (for headless Pi / background service / concurrent apps)")
    p.add_argument("--stride",             type=int, default=1,
                    help="Inference stride: run YOLO detection every Nth frame (default: 1; set 2-3 on Pi 4)")
    p.add_argument("--distance-interval",  type=float, default=config.DEFAULT_DISTANCE_INTERVAL_M,
                    help=f"Geodesic distance in meters between frame captures (default: {config.DEFAULT_DISTANCE_INTERVAL_M}m; 0 for continuous)")
    p.add_argument("--gnss-timeout",       type=float, default=config.GNSS_LOST_TIMEOUT_SEC,
                    help=f"Grace/scan period in seconds on GNSS loss before falling back to clock (default: {config.GNSS_LOST_TIMEOUT_SEC}s)")
    p.add_argument("--fallback-interval",  type=float, default=config.TIME_FALLBACK_INTERVAL_SEC,
                    help=f"Time interval in seconds between frame captures during clock fallback (default: {config.TIME_FALLBACK_INTERVAL_SEC}s)")
    p.add_argument("--trigger-mode",       choices=["auto", "distance", "clock", "continuous"], default="auto",
                    help="Trigger mode: 'auto' (10m distance with 15s GNSS loss / 30s clock fallback), 'distance', 'clock', or 'continuous'")
    p.add_argument("--save-all-triggers",  action="store_true", default=False,
                    help="Save every triggered frame even if no litter was detected (survey mode)")
    return p.parse_args()



# ---------------------------------------------------------------------------
# PHASE 2 — Detection loop
# ---------------------------------------------------------------------------

def run_detection(cap, model, args, poly_pts, rect_aod, tz, location_text=None):
    saved_tracks  = set()
    frame_number  = 0
    paused        = False
    paused_frame  = None
    last_result   = None
    dev_id        = config.load_device_id()

    has_gui = not args.headless
    if has_gui and sys.platform != "win32":
        if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
            print("[GUI] No graphical display detected in environment. Automatically switching to headless mode.")
            has_gui = False

    mode_str = "HEADLESS" if not has_gui else "GUI"
    print(f"DETECTION PHASE started [{mode_str}] | Device ID: {dev_id} | Trigger: {args.trigger_mode} "
          f"({args.distance_interval}m distance / {args.fallback_interval}s clock fallback) | q = quit   t = pause/resume")

    # ---- Trigger state initialization ----
    init_gnss = gnss_sensor.read()
    rtc_snap  = rtc_sensor.read()
    init_has_fix = bool(init_gnss.get("fix") and init_gnss.get("latitude") is not None and init_gnss.get("longitude") is not None)

    if args.trigger_mode == "continuous" or args.distance_interval <= 0:
        op_mode = "CONTINUOUS"
    elif args.trigger_mode == "clock":
        op_mode = "TIME_FALLBACK"
    else:
        # "auto" or "distance"
        op_mode = "GNSS_DISTANCE" if init_has_fix else "TIME_FALLBACK"

    curr_lat = init_gnss.get("latitude") if init_has_fix else None
    curr_lon = init_gnss.get("longitude") if init_has_fix else None
    last_trigger_lat = curr_lat
    last_trigger_lon = curr_lon
    base_lat = curr_lat
    base_lon = curr_lon

    total_distance          = 0.0
    distance_since_trigger  = 0.0
    trigger_count           = 0
    gnss_loss_start_time    = None
    last_fallback_time      = time.monotonic()
    manual_trigger_flag     = False

    if op_mode == "GNSS_DISTANCE":
        print(f"[TRIGGER] Active Mode: GNSS_DISTANCE (10m interval) anchored at base ({base_lat:.8f}, {base_lon:.8f})")
    elif op_mode == "TIME_FALLBACK":
        print(f"[TRIGGER] Active Mode: TIME_FALLBACK ({args.fallback_interval}s interval). Background GNSS scan active.")
    else:
        print("[TRIGGER] Active Mode: CONTINUOUS (per-frame inference)")

    try:
        while cap.isOpened():

            # ---- Pause hold ----
            if paused and paused_frame is not None:
                if has_gui:
                    pf = paused_frame.copy()
                    cv2.putText(pf, "PAUSED  (press T to resume)",
                                (10, pf.shape[0] // 2),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 80, 255), 3)
                    cv2.imshow(WINDOW, pf)
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord("t"):
                        paused = False
                        print("Resumed.")
                    elif key == ord("q"):
                        print("Quit by user.")
                        return saved_tracks
                else:
                    time.sleep(0.1)
                continue

            # ---- Read live camera frame ----
            ok, frame = cap.read()
            if not ok:
                if args.webcam:
                    time.sleep(0.01)
                    continue
                break

            frame_number += 1
            now = time.monotonic()

            # ---- Live GNSS & RTC Sensors ----
            live_gnss = gnss_sensor.read()
            rtc_snap  = rtc_sensor.read()
            has_fix = bool(live_gnss.get("fix") and live_gnss.get("latitude") is not None and live_gnss.get("longitude") is not None)
            if has_fix:
                curr_lat = live_gnss["latitude"]
                curr_lon = live_gnss["longitude"]

            if args.trigger_mode in ("auto", "distance"):
                if has_fix:
                    if op_mode == "TIME_FALLBACK" or last_trigger_lat is None:
                        # Reconnection / initial lock: re-anchor base location!
                        op_mode = "GNSS_DISTANCE"
                        base_lat = curr_lat
                        base_lon = curr_lon
                        last_trigger_lat = curr_lat
                        last_trigger_lon = curr_lon
                        gnss_loss_start_time = None
                        print(f"\n[GNSS REGAINED] 3D Fix re-established @ ({curr_lat:.8f}, {curr_lon:.8f})! "
                              f"Re-anchoring as new base point and resuming {args.distance_interval:.1f}m distance triggers.")
                    elif gnss_loss_start_time is not None:
                        # Recovered within the 15s scan grace period
                        gnss_loss_start_time = None
                else:
                    # Fix lost or unavailable
                    if op_mode == "GNSS_DISTANCE":
                        if gnss_loss_start_time is None:
                            gnss_loss_start_time = now
                            print(f"\n[GNSS LOST] Signal lost. Scanning for GNSS reconnection (grace period: {args.gnss_timeout:.0f}s)...")
                        elif (now - gnss_loss_start_time) >= args.gnss_timeout:
                            op_mode = "TIME_FALLBACK"
                            last_fallback_time = now
                            print(f"\n[TRIGGER FALLBACK] GNSS lost for >{args.gnss_timeout:.0f}s without reconnection. "
                                  f"Falling back to clock-based trigger (capturing every {args.fallback_interval:.0f}s). "
                                  f"Background GNSS scan continues...")

            # ---- Evaluate Trigger Condition ----
            trigger_fired = False
            trigger_reason = ""

            if manual_trigger_flag:
                trigger_fired = True
                trigger_reason = "MANUAL_TRIGGER"
                manual_trigger_flag = False

            elif op_mode == "CONTINUOUS":
                skip_inference = (args.stride > 1 and (frame_number % args.stride != 0) and last_result is not None)
                if not skip_inference:
                    trigger_fired = True
                    trigger_reason = "CONTINUOUS"

            elif op_mode == "GNSS_DISTANCE":
                if has_fix and last_trigger_lat is not None and last_trigger_lon is not None:
                    dist = gnss_sensor.haversine_distance_m(last_trigger_lat, last_trigger_lon, curr_lat, curr_lon)
                    distance_since_trigger = dist
                    if dist >= args.distance_interval:
                        trigger_fired = True
                        trigger_reason = f"DISTANCE ({dist:.2f}m >= {args.distance_interval:.1f}m)"
                        total_distance += dist
                        last_trigger_lat = curr_lat
                        last_trigger_lon = curr_lon
                        distance_since_trigger = 0.0

            elif op_mode == "TIME_FALLBACK":
                elapsed = now - last_fallback_time
                if elapsed >= args.fallback_interval:
                    trigger_fired = True
                    trigger_reason = f"CLOCK_FALLBACK ({elapsed:.1f}s >= {args.fallback_interval:.0f}s, scanning GNSS)"
                    last_fallback_time = now

            # ---- Execute Inference & Event Logging on Trigger ----
            save_frame     = False
            save_canvas    = frame.copy()
            detected_items = []

            if trigger_fired:
                trigger_count += 1
                print(f"\n>>> [TRIGGER #{trigger_count}] Fired: {trigger_reason}")

                # Triple blink Orange LED on GPIO 25 to indicate frame capture
                leds_sensor.notify_capture()

                # In live webcam mode, flush driver buffer to guarantee live real-time frame
                if args.webcam:
                    for _ in range(3):
                        cap.grab()
                    ok_fresh, fresh_frame = cap.read()
                    if ok_fresh:
                        frame = fresh_frame
                        save_canvas = frame.copy()


                # Run model inference on triggered frame
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=args.conf,
                    verbose=False,
                )
                result = results[0]
                last_result = result

                if result.boxes is not None:
                    for box in result.boxes:
                        if box.id is None:
                            continue

                        track_id        = int(box.id.item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf_score      = float(box.conf[0])
                        cls_id          = int(box.cls[0])
                        cls_name        = model.names.get(cls_id, str(cls_id))

                        if poly_pts:
                            in_aod = (polygon_overlap((x1, y1, x2, y2),
                                                      poly_pts, frame.shape)
                                      >= args.overlap_threshold)
                        else:
                            in_aod = (rect_overlap((x1, y1, x2, y2), rect_aod)
                                      >= args.overlap_threshold)

                        detected_items.append({
                            "track_id": track_id,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(conf_score, 4),
                            "bbox": [x1, y1, x2, y2],
                            "in_aod": in_aod
                        })

                        if in_aod:
                            continue

                        label = f"{cls_name} #{track_id}  {conf_score:.2f}"
                        cv2.rectangle(save_canvas, (x1, y1), (x2, y2), (0, 230, 0), 3)
                        cv2.putText(save_canvas, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2)

                        if track_id not in saved_tracks:
                            saved_tracks.add(track_id)
                            save_frame = True

                if args.save_all_triggers:
                    save_frame = True

                # ---- Save event (Exact reference styling: non-concealing canvas extension banner) ----
                if save_frame:
                    # 1. RTC time extraction (ISO format with 'T' matching reference image)
                    rtc_data = rtc_snap
                    if rtc_data.get("valid") and rtc_data.get("timestamp"):
                        rtc_display = rtc_data["timestamp"]
                        rtc_src = rtc_sensor.sync_source.upper()
                    else:
                        rtc_display = datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S")
                        rtc_src = "SYS"

                    # 2. GNSS coordinates extraction
                    lat  = curr_lat if has_fix else None
                    lon  = curr_lon if has_fix else None
                    spd  = live_gnss.get("speed_kmh") or 0.0
                    sats = live_gnss.get("satellites", 0)

                    if has_fix and lat is not None and lon is not None:
                        gps_display = f"GNSS: {lat:.8f}, {lon:.8f}"
                        gnss_col = (0, 255, 0)      # Bright Green (exact match)
                        loc_src = "navcast_gnss"
                    elif args.latitude is not None and args.longitude is not None:
                        lat = args.latitude
                        lon = args.longitude
                        gps_display = f"GNSS: {lat:.8f}, {lon:.8f}"
                        gnss_col = (0, 255, 0)
                        loc_src = "manual"
                    else:
                        lat = None
                        lon = None
                        loc_src = "none"
                        if op_mode == "TIME_FALLBACK":
                            gps_display = "GNSS: NO FIX (TIME TRIGGER)"
                            gnss_col = (0, 165, 255)
                        else:
                            gps_display = "GNSS: NO FIX"
                            gnss_col = (0, 0, 255)

                    filename = os.path.join(args.output, f"Frame_{frame_number}.jpg")
                    strip_h  = 70
                    w_img    = save_canvas.shape[1]

                    # Non-destructive canvas extension banner appended BELOW the image (0% image covered)
                    banner = np.zeros((strip_h, w_img, 3), dtype=np.uint8)
                    cv2.line(banner, (0, 0), (w_img, 0), (50, 50, 50), 1)

                    # Line 1: RTC timestamp in bright yellow (0, 255, 255)
                    cv2.putText(banner, f"RTC: {rtc_display}",
                                (14, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                    # Line 2: GNSS coordinates in bright green (0, 255, 0)
                    cv2.putText(banner, gps_display,
                                (14, 54),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, gnss_col, 2, cv2.LINE_AA)

                    # Vertically stack image and banner so frame data is NEVER concealed
                    final_frame = np.vstack([save_canvas, banner])
                    sidecar_filename = os.path.splitext(filename)[0] + ".json"

                    # Write telemetry sidecar JSON
                    sidecar_data = {
                        "frame_number": frame_number,
                        "image_file": os.path.basename(filename),
                        "device_id": dev_id,
                        "trigger_mode": op_mode,
                        "trigger_reason": trigger_reason,
                        "timestamp": rtc_display,
                        "epoch_ms": rtc_data.get("epoch") or int(datetime.now(tz).timestamp() * 1000),
                        "rtc": {
                            "valid": bool(rtc_data.get("valid")),
                            "source": rtc_src,
                            "hw_source": rtc_data.get("hw_source", "software")
                        },
                        "gnss": {
                            "fix": bool(has_fix),
                            "status": live_gnss.get("status", "NO_DATA"),
                            "source": loc_src,
                            "latitude": lat,
                            "longitude": lon,
                            "altitude_m": live_gnss.get("altitude_m"),
                            "speed_kmh": live_gnss.get("speed_kmh"),
                            "course_deg": live_gnss.get("course_deg"),
                            "satellites": sats,
                            "hdop": live_gnss.get("hdop")
                        },
                        "detections": detected_items
                    }

                    # Enqueue for asynchronous background writing (zero save latency impact)
                    _save_queue.put((filename, final_frame, sidecar_filename, sidecar_data))
                    print(f"[EVENT] Saved {filename} ({trigger_reason}, {len(detected_items)} detections)")
                else:
                    print(f"[TRIGGER #{trigger_count}] Processed — Clean frame (0 new litter detections).")

            # ---- Update Orange Status LED (GPIO 25 / Pin 22) ----
            # Solid ON when functional; Constant blinking when scanning for GNSS
            if has_fix and op_mode == "GNSS_DISTANCE":
                leds_sensor.set_functional()
            else:
                leds_sensor.set_scanning_gnss()


            # ---- Viewfinder Display & HUD (in GUI mode) ----
            if has_gui:
                display = frame.copy()
                if poly_pts:
                    draw_polygon_aod(display, poly_pts)
                else:
                    draw_rect_aod(display, rect_aod)

                # Draw recent detections on preview
                if last_result is not None and last_result.boxes is not None:
                    for box in last_result.boxes:
                        if box.id is None:
                            continue
                        tid = int(box.id.item())
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        bconf = float(box.conf[0])
                        bcls = int(box.cls[0])
                        bname = model.names.get(bcls, str(bcls))
                        in_a = False
                        if poly_pts:
                            in_a = (polygon_overlap((bx1, by1, bx2, by2), poly_pts, frame.shape) >= args.overlap_threshold)
                        else:
                            in_a = (rect_overlap((bx1, by1, bx2, by2), rect_aod) >= args.overlap_threshold)

                        col = (80, 80, 80) if in_a else (0, 230, 0)
                        cv2.rectangle(display, (bx1, by1), (bx2, by2), col, 2)
                        blabel = f"{bname} #{tid}  {bconf:.2f}"
                        cv2.putText(display, blabel, (bx1 + 2, max(15, by1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

                # Timestamp HUD
                if rtc_snap.get("valid") and rtc_snap.get("timestamp"):
                    ts = rtc_snap["timestamp"].replace("T", " ")
                    rtc_hud_src = rtc_sensor.sync_source.upper()
                else:
                    ts = datetime.now(tz).strftime("%d-%m-%Y  %H:%M:%S")
                    rtc_hud_src = "SYS"

                # GNSS Status HUD
                if has_fix:
                    gnss_hud = f"GNSS: 3D FIX ({live_gnss.get('satellites', 0)} sats)"
                    gnss_hud_col = (0, 255, 0)
                elif live_gnss.get("data_received"):
                    gnss_hud = f"GNSS: CONNECTED (No Fix - {live_gnss.get('satellites', 0)} sats)"
                    gnss_hud_col = (0, 215, 255)
                else:
                    gnss_hud = f"GNSS: {live_gnss.get('status', 'SEARCHING')}"
                    gnss_hud_col = (0, 165, 255)

                # Mode HUD
                if op_mode == "GNSS_DISTANCE":
                    mode_hud = f"TRIG: 10m GNSS | Moved: {distance_since_trigger:.1f}m / {args.distance_interval:.1f}m | Total: {total_distance:.1f}m"
                    mode_hud_col = (0, 255, 255)
                elif op_mode == "TIME_FALLBACK":
                    next_sec = max(0.0, args.fallback_interval - (now - last_fallback_time))
                    mode_hud = f"TRIG: CLOCK FALLBACK | Next in: {next_sec:.0f}s | GNSS: SCANNING"
                    mode_hud_col = (0, 165, 255)
                else:
                    mode_hud = "TRIG: CONTINUOUS"
                    mode_hud_col = (200, 200, 200)

                src = f"Webcam #{args.camera_id}" if args.webcam else os.path.basename(args.video)

                cv2.putText(display, f"{ts} [{rtc_hud_src}]", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(display, gnss_hud, (10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, gnss_hud_col, 2)
                cv2.putText(display, mode_hud, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_hud_col, 2)
                cv2.putText(display, f"Captures: {trigger_count} | Events saved: {len(saved_tracks)} | {src}", (10, 106),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)
                cv2.putText(display, "q=quit  t=pause  space/c=manual capture",
                            (10, display.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

                try:
                    cv2.imshow(WINDOW, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("Quit by user.")
                        break
                    elif key == ord("t"):
                        paused       = True
                        paused_frame = display.copy()
                        print("Paused. Press T in the window to resume.")
                    elif key in (ord(" "), ord("c")):
                        manual_trigger_flag = True
                except cv2.error as e:
                    print(f"[GUI] Display error ({e}). Switching to headless mode.")
                    has_gui = False
            else:
                # In headless mode: sleep briefly to avoid pegging CPU while waiting for distance
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")


    return saved_tracks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Initialize Subsystems (LEDs, RTC & NavCast GNSS) ----
    print("\n[INIT] Initializing Orange Status LED (BCM GPIO 25 / Physical Pin 22)...")
    leds_sensor.init()
    leds_sensor.set_scanning_gnss()  # Constant blinking while scanning for GNSS

    print("\n[INIT] Initializing RTC (DS3231)...")
    rtc_ok = rtc_sensor.init()
    print(f"[RTC] Status: {'OK (Hardware DS3231)' if rtc_ok else 'FALLBACK (System Clock)'} [Source: {rtc_sensor.sync_source.upper()}]")

    print("\n[INIT] Initializing GNSS (NavCast TCP NMEA)...")
    gnss_ok = gnss_sensor.init()
    print(f"[GNSS] NavCast background driver initialized (auto-detect: {config.NAVCAST_AUTO_DETECT})")

    # Scan for initial base GNSS 3D fix for up to args.gnss_timeout (default: 15s)
    print(f"\n[BOOT] Scanning for base GNSS starting location (waiting up to {args.gnss_timeout:.0f}s for 3D fix)...")
    t_gnss_start = time.time()
    initial_fix = False
    while time.time() - t_gnss_start < args.gnss_timeout:
        snap = gnss_sensor.read()
        if snap.get("fix") and snap.get("latitude") is not None and snap.get("longitude") is not None:
            initial_fix = True
            leds_sensor.set_functional()  # Solid ON when system is functional!
            print(f"[BOOT] Base location locked in {time.time() - t_gnss_start:.1f}s: "
                  f"{snap['latitude']:.8f}, {snap['longitude']:.8f} ({snap.get('satellites', 0)} sats).")
            print(f"[BOOT] Initializing {args.distance_interval:.1f}m GNSS distance-trigger loop.")
            break
        elif snap.get("data_received") and int(time.time() - t_gnss_start) % 3 == 0:
            print(f"[BOOT] NavCast connected ({snap.get('satellites', 0)} sats in view), waiting for 3D fix...")
        time.sleep(0.5)

    if not initial_fix:
        print(f"[BOOT] GNSS 3D fix not acquired within {args.gnss_timeout:.0f}s.")
        print(f"[BOOT] Falling back to clock-based trigger (capturing every {args.fallback_interval:.0f}s).")
        print(f"[BOOT] Background GNSS auto-reconnection active — will switch to {args.distance_interval:.1f}m distance mode as soon as fix is acquired.")

    # Start background RTC sync thread
    sync_thread = threading.Thread(
        target=periodic_sync_loop,
        kwargs={"interval_hours": config.RTC_RESYNC_INTERVAL_HOURS},
        daemon=True,
        name="rtc-periodic-sync",
    )
    sync_thread.start()

    # Start asynchronous disk writer thread
    save_thread = threading.Thread(target=_save_worker, daemon=True, name="event-save-worker")
    save_thread.start()

    # Headless mode overrides
    if args.headless:
        args.draw_aod = False
        args.show = False

    if not os.path.isfile(args.weights):
        print(f"[WARN] Weights file not found: {args.weights}")

    # Open capture source
    if args.webcam:
        print(f"Opening webcam (camera id: {args.camera_id}) ...")
        cap = cv2.VideoCapture(args.camera_id)
        # Cap webcam hardware resolution on Pi to save USB bandwidth and resize overhead
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not args.headless:
            args.show = True
    else:
        if not os.path.isfile(args.video):
            leds_sensor.set_error(f"Video file not found: {args.video}")
            time.sleep(2.0)
            sys.exit(f"Video file not found: {args.video}")
        print(f"Opening video: {args.video}")
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        src = f"camera {args.camera_id}" if args.webcam else args.video
        print(f"[ERROR] Could not open source: {src}")
        # Heartbeat blink on hardware error
        leds_sensor.set_error(f"Could not open source: {src}")
        time.sleep(2.0)
        gnss_sensor.stop()
        leds_sensor.close()
        sys.exit(1)


    os.makedirs(args.output, exist_ok=True)
    try:
        tz = ZoneInfo(args.timezone)
    except Exception:
        tz = config.get_timezone_obj()
    location_text = (
        f"GPS: {args.latitude:.6f}, {args.longitude:.6f}"
        if (args.latitude is not None and args.longitude is not None)
        else "GPS: LIVE (GNSS / NavCast)"
    )
    rect_aod      = tuple(args.aod)

    # ---- Phase 1: Setup (polygon drawing) ----
    poly_pts = None
    if args.draw_aod and not args.headless:
        drawn = run_setup_phase(cap)
        poly_pts = drawn if len(drawn) >= 3 else None
        if poly_pts:
            print(f"Polygon AoD confirmed ({len(poly_pts)} vertices). Starting detection...")
        else:
            print("No polygon AoD. Starting detection with rectangular fallback...")
    else:
        if not args.headless:
            print("Skipping setup phase. Using rectangular AoD.")
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        else:
            print("Running in HEADLESS mode. Using rectangular AoD.")

    # ---- Load model ----
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Signal system is ready for detection
    leds_sensor.set_system_ready()

    # ---- Phase 2: Detection ----
    saved = set()
    try:
        saved = run_detection(cap, model, args, poly_pts, rect_aod, tz, location_text)
    except Exception as exc:
        print(f"[FATAL ERROR] Detection terminated with error: {exc}")
        leds_sensor.set_error(str(exc))
        time.sleep(2.0)
        raise
    finally:
        print("\n[SHUTDOWN] Waiting for pending event writes to complete...")

        _save_queue.join()
        _save_queue.put(None)
        print("[SHUTDOWN] Stopping NavCast GNSS reader thread & closing LEDs...")
        gnss_sensor.stop()
        leds_sensor.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Cleaned up camera, LEDs, and windows.")

    print(f"Finished. Total events saved: {len(saved)}")


if __name__ == "__main__":
    main()
