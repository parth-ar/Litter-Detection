# Review App — how to run it

## 1. Install (same venv as before)

```bash
pip install ultralytics opencv-python numpy
```

## 2. Point it at your ~100 images

If you haven't run inference yet, just give it your weights directly —
it'll predict live, frame by frame, as you review:

```bash
python review_app.py \
    --images-dir raw_frames/ \
    --weights best.pt \
    --classes "cigarette,plastic_bottle,wrapper,can,other" \
    --output active_learning_batch1 \
    --conf 0.15
```

`--conf 0.15` is intentionally low — you want to *see* the model's borderline
guesses so you can reject them, not have them hidden. Match `--classes`
to your training data.yaml order exactly (index 0 must be the same class
you trained as index 0).

If you already have prediction `.txt` files (e.g. from `yolo predict
save_txt=True save_conf=True`), use `--preds-dir` instead of `--weights`
and it'll skip live inference.

## 3. Review loop, per image

- Boxes appear with class name + confidence.
- **Correct as-is?** Press `s` or Enter → saved, moves to next image.
- **Totally wrong / not litter?** Press `n` or Space → skipped, not saved,
  moves to next image.
- **Box is right but class is wrong?** Click the box (it highlights yellow
  with corner handles), then press the digit for the correct class.
- **Box position/size is off?** Drag the middle to move it, drag a corner
  to resize it.
- **Model missed something?** Left-click-drag on empty space to draw a new
  box. It uses whatever the "active class" is (shown top-left) — set that
  first by pressing a digit while nothing is selected.
- **False positive box?** Right-click it to delete.
- **Messed something up?** `z` undoes the last box you added, `u` resets
  the whole frame back to the model's original predictions.
- `q` or `Esc` quits and saves your progress — rerun the same command
  later and it picks up where you left off (tracked in
  `active_learning_batch1/review_state.json`).

## 4. What you get out

```
active_learning_batch1/
├── images/            # accepted frames only
├── labels/             # matching YOLO-format .txt (class xc yc w h)
└── review_state.json   # which files are accepted/rejected, for resuming
```

This folder is now ready to merge into your training set for the retrain
step (Phase 3) — treat it as a new data source alongside your original
TACO-derived dataset, not a replacement for it.

## Tuning notes

- If you find yourself rejecting almost every frame, that's a signal the
  model needs more diverse training data, not just corrections — worth
  revisiting before sinking time into review.
- If most rejects are one class getting confused with another, you can
  triage faster: review just that class's frames first.
- Keep batches small (this ~100-image batch) so you can retrain and
  re-evaluate quickly rather than accumulating a huge uncurated backlog.
