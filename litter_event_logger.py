"""
Litter Detection Event Logger
------------------------------
Runs a YOLO model with ByteTrack on a video or live webcam feed, ignores
detections inside a polygon Area of Disinterest (AoD), and saves one
annotated frame per newly-seen tracked object.

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
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Defaults  (edit here to run without CLI args in Antigravity)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS   = r"weights\best.pt"
DEFAULT_VIDEO     = r"test vid/trash stock.webm"
DEFAULT_OUTPUT    = r"runs/EventLogger"
DEFAULT_CAMERA_ID = 0
DEFAULT_WEBCAM    = True   # False -> use video file
DEFAULT_DRAW_AOD  = True   # False -> skip polygon setup, use rect AoD

DEFAULT_AOD               = (500, 1000, 1200, 3500)  # x1 y1 x2 y2 rect fallback
DEFAULT_OVERLAP_THRESHOLD = 0.50
DEFAULT_LATITUDE          = 18.52043025
DEFAULT_LONGITUDE         = 73.85674345
DEFAULT_TIMEZONE          = "Asia/Kolkata"
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
    p = argparse.ArgumentParser(description="Litter detection event/frame logger")
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
    p.add_argument("--latitude",           type=float, default=DEFAULT_LATITUDE)
    p.add_argument("--longitude",          type=float, default=DEFAULT_LONGITUDE)
    p.add_argument("--timezone",           default=DEFAULT_TIMEZONE)
    p.add_argument("--show",               action="store_true",
                   help="Show preview window (auto-on for webcam/draw-aod)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# PHASE 2 — Detection loop
# ---------------------------------------------------------------------------

def run_detection(cap, model, args, poly_pts, rect_aod, tz, location_text):
    saved_tracks  = set()
    frame_number  = 0
    paused        = False
    paused_frame  = None

    print("DETECTION PHASE started  |  q = quit   t = pause/resume")

    try:
        while cap.isOpened():

            # ---- Pause hold ----
            if paused and paused_frame is not None:
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
                continue

            # ---- Read frame ----
            ok, frame = cap.read()
            if not ok:
                if args.webcam:
                    continue
                break

            frame_number += 1

            # ---- Tracker ----
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                verbose=False,
            )
            result = results[0]

            # ---- Build display ----
            display = frame.copy()
            if poly_pts:
                draw_polygon_aod(display, poly_pts)
            else:
                draw_rect_aod(display, rect_aod)

            save_frame  = False
            save_canvas = frame.copy()

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

                    # Every detection drawn on viewfinder
                    color = (80, 80, 80) if in_aod else (0, 230, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} #{track_id}  {conf_score:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display,
                                  (x1, y1 - th - 8), (x1 + tw + 4, y1),
                                  color, -1)
                    cv2.putText(display, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    if in_aod or track_id in saved_tracks:
                        continue
                    saved_tracks.add(track_id)
                    save_frame = True

                    cv2.rectangle(save_canvas, (x1, y1), (x2, y2), (0, 230, 0), 3)
                    cv2.putText(save_canvas, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2)

            # ---- HUD ----
            ts = datetime.now(tz).strftime("%d-%m-%Y  %H:%M:%S")
            src = (f"Webcam #{args.camera_id}"
                   if args.webcam else os.path.basename(args.video))
            cv2.putText(display, ts, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Events saved: {len(saved_tracks)}", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
            cv2.putText(display, src, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(display, "q=quit  t=pause",
                        (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # ---- Save event ----
            if save_frame:
                ts_stamp = datetime.now(tz).strftime("%d-%m-%Y %H:%M:%S")
                filename  = os.path.join(args.output, f"Frame_{frame_number}.jpg")
                strip_h   = 80
                h_img     = save_canvas.shape[0]
                cv2.rectangle(save_canvas,
                              (0, h_img - strip_h),
                              (save_canvas.shape[1], h_img),
                              (255, 255, 255), -1)
                cv2.putText(save_canvas, f"Time: {ts_stamp}",
                            (15, h_img - strip_h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                cv2.putText(save_canvas, location_text,
                            (15, h_img - strip_h + 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                cv2.imwrite(filename, save_canvas)
                print(f"[EVENT] Saved {filename}")

            # ---- Show + key handling ----
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit by user.")
                break
            elif key == ord("t"):
                paused       = True
                paused_frame = display.copy()
                print("Paused. Press T in the window to resume.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    return saved_tracks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.weights):
        sys.exit(f"Weights file not found: {args.weights}")

    # Open capture source
    if args.webcam:
        print(f"Opening webcam (camera id: {args.camera_id}) ...")
        cap = cv2.VideoCapture(args.camera_id)
        args.show = True
    else:
        if not os.path.isfile(args.video):
            sys.exit(f"Video file not found: {args.video}")
        print(f"Opening video: {args.video}")
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        src = f"camera {args.camera_id}" if args.webcam else args.video
        sys.exit(f"Could not open source: {src}")

    os.makedirs(args.output, exist_ok=True)
    tz            = ZoneInfo(args.timezone)
    location_text = f"Lat: {args.latitude:.6f}   Lon: {args.longitude:.6f}"
    rect_aod      = tuple(args.aod)

    # ---- Phase 1: Setup (polygon drawing) ----
    poly_pts = None
    if args.draw_aod:
        drawn = run_setup_phase(cap)
        poly_pts = drawn if len(drawn) >= 3 else None
        if poly_pts:
            print(f"Polygon AoD confirmed ({len(poly_pts)} vertices). Starting detection...")
        else:
            print("No polygon AoD. Starting detection with rectangular fallback...")
    else:
        print("Skipping setup phase. Using rectangular AoD.")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # ---- Load model ----
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # ---- Phase 2: Detection ----
    saved = run_detection(cap, model, args, poly_pts, rect_aod, tz, location_text)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. Total events saved: {len(saved)}")


if __name__ == "__main__":
    main()
