"""
Active-Learning Review App
---------------------------
Walks through a folder of images, shows the model's predicted boxes, and
lets you correct them with the mouse/keyboard. Confirmed frames are saved
in YOLO format (images/ + labels/) ready to merge into training data.

Two ways to feed it predictions:
  1. Point --weights at a .pt file -> it runs inference itself, per image.
  2. Point --preds-dir at a folder of YOLO-format .txt files (e.g. output
     of `yolo predict save_txt=True save_conf=True`) -> it reuses those.

Progress is saved to review_state.json in --output, so you can quit and
resume later without re-reviewing frames you already handled.

CONTROLS (shown on-screen too):
  Left-drag on empty space   Draw a new box (uses the active class)
  Left-drag inside a box     Move the box
  Left-drag near a corner    Resize the box
  Right-click a box          Delete that box
  0-9                        Set active class for NEW boxes
  Shift + 0-9  (see note)    Reclass the *last clicked* box
  s / Enter                  Accept frame -> save image + label, next
  n / Space                  Reject frame (skip, not saved), next
  z                          Undo last added box
  u                          Reset frame to original model predictions
  q / Esc                    Quit (progress is saved)

Note: OpenCV can't reliably detect Shift+digit in all backends, so instead:
  click a box once to "select" it (highlighted yellow), then press a digit
  0-9 to reclass just that box. Click empty space to deselect (back to
  "set active class for new boxes" mode).

Usage:
    python review_app.py --images-dir raw_frames/ --weights best.pt \
        --classes "cigarette,plastic_bottle,wrapper,can,other" \
        --output active_learning_batch1
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

CORNER_GRAB_PX = 10
BOX_HIT_MARGIN = 4
COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 255, 0), (0, 128, 0),
    (128, 128, 255), (255, 0, 0),
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    import datetime
    _today = datetime.date.today().strftime("%Y_%m_%d")
    p = argparse.ArgumentParser(description="Active-learning correction/review app")
    p.add_argument("--images-dir", default="test images",
                   help="Folder of images to review (default: 'test images')")
    p.add_argument("--weights", default="weights/best.pt",
                   help="YOLO .pt to run inference with (default: weights/best.pt)")
    p.add_argument("--preds-dir", default=None,
                   help="Folder of pre-computed YOLO-format .txt predictions (optional)")
    p.add_argument("--classes",
                   default="litter",
                   help="Comma-separated class names in index order (default: cigarette,plastic_bottle,wrapper,can,other)")
    p.add_argument("--output", default=os.path.join("reviewed_output", f"batch_{_today}"),
                   help="Where to write reviewed images/ + labels/ (default: reviewed_output/batch_YYYY_MM_DD)")
    p.add_argument("--conf", type=float, default=0.15,
                   help="Min confidence to show when running inference live (keep low; you decide what's real)")
    return p.parse_args()


def load_yolo_txt(path, img_w, img_h):
    """Read a YOLO-format .txt (5 or 6 columns) into pixel-space boxes."""
    boxes = []
    if not os.path.isfile(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            boxes.append({"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})
    return boxes


def save_yolo_txt(path, boxes, img_w, img_h):
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{b['cls']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


class ReviewSession:
    def __init__(self, class_names):
        self.class_names = class_names
        self.boxes = []
        self.orig_boxes = []
        self.active_cls = 0
        self.selected_idx = None
        self.drag_mode = None  # 'new' | 'move' | 'resize'
        self.drag_corner = None  # which corner index for resize
        self.drag_start = None
        self.drag_box_start = None
        self.img_w = 0
        self.img_h = 0

    def load(self, boxes, img_w, img_h):
        self.boxes = [dict(b) for b in boxes]
        self.orig_boxes = [dict(b) for b in boxes]
        self.img_w = img_w
        self.img_h = img_h
        self.selected_idx = None

    def reset_to_original(self):
        self.boxes = [dict(b) for b in self.orig_boxes]
        self.selected_idx = None

    def hit_box(self, x, y):
        """Return index of topmost box containing (x,y), or None."""
        for i in reversed(range(len(self.boxes))):
            b = self.boxes[i]
            if (b["x1"] - BOX_HIT_MARGIN <= x <= b["x2"] + BOX_HIT_MARGIN and
                    b["y1"] - BOX_HIT_MARGIN <= y <= b["y2"] + BOX_HIT_MARGIN):
                return i
        return None

    def hit_corner(self, box, x, y):
        corners = {
            0: (box["x1"], box["y1"]), 1: (box["x2"], box["y1"]),
            2: (box["x1"], box["y2"]), 3: (box["x2"], box["y2"]),
        }
        for idx, (cx, cy) in corners.items():
            if abs(cx - x) <= CORNER_GRAB_PX and abs(cy - y) <= CORNER_GRAB_PX:
                return idx
        return None

    def mouse_callback(self, event, x, y, flags, param):
        x = max(0, min(self.img_w - 1, x))
        y = max(0, min(self.img_h - 1, y))

        if event == cv2.EVENT_RBUTTONDOWN:
            idx = self.hit_box(x, y)
            if idx is not None:
                del self.boxes[idx]
                self.selected_idx = None
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self.hit_box(x, y)
            if idx is not None:
                box = self.boxes[idx]
                corner = self.hit_corner(box, x, y)
                self.selected_idx = idx
                if corner is not None:
                    self.drag_mode = "resize"
                    self.drag_corner = corner
                else:
                    self.drag_mode = "move"
                self.drag_start = (x, y)
                self.drag_box_start = dict(box)
            else:
                self.selected_idx = None
                self.drag_mode = "new"
                self.drag_start = (x, y)
                self.boxes.append({"cls": self.active_cls, "x1": x, "y1": y, "x2": x, "y2": y, "conf": None})
                self.selected_idx = len(self.boxes) - 1

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_mode == "new" and self.selected_idx is not None:
                b = self.boxes[self.selected_idx]
                b["x1"], b["y1"] = self.drag_start
                b["x2"], b["y2"] = x, y
            elif self.drag_mode == "move" and self.selected_idx is not None:
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                b = self.boxes[self.selected_idx]
                s = self.drag_box_start
                b["x1"], b["y1"], b["x2"], b["y2"] = s["x1"] + dx, s["y1"] + dy, s["x2"] + dx, s["y2"] + dy
            elif self.drag_mode == "resize" and self.selected_idx is not None:
                b = self.boxes[self.selected_idx]
                if self.drag_corner in (0, 2):
                    b["x1"] = x
                else:
                    b["x2"] = x
                if self.drag_corner in (0, 1):
                    b["y1"] = y
                else:
                    b["y2"] = y

        elif event == cv2.EVENT_LBUTTONUP:
            if self.selected_idx is not None:
                b = self.boxes[self.selected_idx]
                b["x1"], b["x2"] = sorted((b["x1"], b["x2"]))
                b["y1"], b["y2"] = sorted((b["y1"], b["y2"]))
                if b["x2"] - b["x1"] < 3 or b["y2"] - b["y1"] < 3:
                    del self.boxes[self.selected_idx]
                    self.selected_idx = None
            self.drag_mode = None
            self.drag_corner = None

    def set_class_key(self, digit):
        if self.selected_idx is not None:
            self.boxes[self.selected_idx]["cls"] = digit
        else:
            self.active_cls = digit

    def undo_last(self):
        if self.boxes:
            self.boxes.pop()
            self.selected_idx = None

    def render(self, frame):
        vis = frame.copy()
        for i, b in enumerate(self.boxes):
            color = COLORS[b["cls"] % len(COLORS)]
            thickness = 3 if i == self.selected_idx else 2
            cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thickness)
            name = self.class_names[b["cls"]] if b["cls"] < len(self.class_names) else str(b["cls"])
            label = name if b["conf"] is None else f"{name} {b['conf']:.2f}"
            cv2.putText(vis, label, (b["x1"], max(15, b["y1"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if i == self.selected_idx:
                for cx, cy in [(b["x1"], b["y1"]), (b["x2"], b["y1"]), (b["x1"], b["y2"]), (b["x2"], b["y2"])]:
                    cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)

        active_name = self.class_names[self.active_cls] if self.active_cls < len(self.class_names) else "?"
        hud_lines = [
            f"Active class [{self.active_cls}]: {active_name}   "
            f"(select a box + digit to reclass just that box)",
            "s/Enter accept | n/Space reject | z undo | u reset | drag=draw/move/resize | right-click=delete | q quit",
        ]
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
        vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)
        for i, line in enumerate(hud_lines):
            cv2.putText(vis, line, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return vis


def main():
    args = parse_args()
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not class_names:
        sys.exit("--classes must list at least one class name")

    if not os.path.isdir(args.images_dir):
        sys.exit(f"Images dir not found: {args.images_dir}")

    out_images = os.path.join(args.output, "images")
    out_labels = os.path.join(args.output, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)
    state_path = os.path.join(args.output, "review_state.json")

    state = {"accepted": [], "rejected": []}
    if os.path.isfile(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
    done = set(state["accepted"]) | set(state["rejected"])

    model = None
    if args.weights:
        from ultralytics import YOLO
        print(f"Loading model: {args.weights}")
        model = YOLO(args.weights)
    elif not args.preds_dir:
        sys.exit("Provide either --weights (to run inference) or --preds-dir (pre-computed predictions)")

    files = sorted(f for f in os.listdir(args.images_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS)
    todo = [f for f in files if f not in done]
    print(f"{len(files)} images total, {len(todo)} left to review, {len(done)} already done.")

    # ── Pre-run inference on all pending images before review starts ──────────
    pred_cache = {}  # fname -> list of box dicts
    if model is not None:
        print(f"Running inference on {len(todo)} images (please wait)...")
        for i, fname in enumerate(todo, 1):
            img_path = os.path.join(args.images_dir, fname)
            frame_tmp = cv2.imread(img_path)
            if frame_tmp is None:
                pred_cache[fname] = []
                continue
            h_tmp, w_tmp = frame_tmp.shape[:2]
            results = model.predict(frame_tmp, conf=args.conf, verbose=False)
            r = results[0]
            boxes_tmp = []
            if r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cls = int(b.cls.item())
                    conf = float(b.conf.item())
                    boxes_tmp.append({"cls": cls, "x1": x1, "y1": y1,
                                      "x2": x2, "y2": y2, "conf": conf})
            pred_cache[fname] = boxes_tmp
            print(f"  [{i}/{len(todo)}] {fname} — {len(boxes_tmp)} detection(s)")
        print("Inference complete. Opening review window...\n")

    session = ReviewSession(class_names)
    window = "Active Learning Review"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, session.mouse_callback)

    idx = 0
    while idx < len(todo):
        fname = todo[idx]
        img_path = os.path.join(args.images_dir, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Skipping unreadable file: {fname}")
            idx += 1
            continue
        h, w = frame.shape[:2]

        if args.preds_dir:
            txt_path = os.path.join(args.preds_dir, os.path.splitext(fname)[0] + ".txt")
            boxes = load_yolo_txt(txt_path, w, h)
        else:
            # Use pre-computed cache; fall back to live inference if somehow missing
            if fname in pred_cache:
                boxes = pred_cache[fname]
            else:
                results = model.predict(frame, conf=args.conf, verbose=False)
                r = results[0]
                boxes = []
                if r.boxes is not None:
                    for b in r.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cls = int(b.cls.item())
                        conf = float(b.conf.item())
                        boxes.append({"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})

        session.load(boxes, w, h)

        accepted_or_rejected = False
        while not accepted_or_rejected:
            vis = session.render(frame)
            cv2.imshow(window, vis)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=2)
                cv2.destroyAllWindows()
                print(f"Progress saved. {len(state['accepted'])} accepted, {len(state['rejected'])} rejected so far.")
                return

            elif key in (ord("s"), 13):
                save_yolo_txt(os.path.join(out_labels, os.path.splitext(fname)[0] + ".txt"),
                               session.boxes, w, h)
                cv2.imwrite(os.path.join(out_images, fname), frame)
                state["accepted"].append(fname)
                print(f"[accept] {fname} ({len(session.boxes)} boxes)")
                accepted_or_rejected = True

            elif key in (ord("n"), ord(" ")):
                state["rejected"].append(fname)
                print(f"[reject] {fname}")
                accepted_or_rejected = True

            elif key == ord("z"):
                session.undo_last()

            elif key == ord("u"):
                session.reset_to_original()

            elif ord("0") <= key <= ord("9"):
                session.set_class_key(key - ord("0"))

        idx += 1
        if idx % 10 == 0:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    cv2.destroyAllWindows()
    print(f"Done. {len(state['accepted'])} accepted, {len(state['rejected'])} rejected -> saved to {args.output}")


if __name__ == "__main__":
    main()
