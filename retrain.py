"""
Active-Learning Fine-tune
--------------------------
Takes a reviewed batch from review_app.py (images/ + labels/, YOLO format,
including empty label files for confirmed-negative frames) and fine-tunes
an existing YOLO model on it.

This is intentionally conservative: low learning rate, few epochs, and an
optional frozen backbone — because you're correcting a model with a small
batch, not training from scratch. Since you don't have your original
dataset anymore, this also guards a bit against catastrophic forgetting
(the model unlearning things it used to know).

What it does:
  1. Splits your reviewed batch into train/val (stratified isn't worth it
     for ~100 images; a random split is fine).
  2. Writes a YOLO-format dataset (images/train, images/val, labels/train,
     labels/val) + data.yaml.
  3. Fine-tunes from --weights using model.train(...).
  4. Prints where the new best.pt landed, and reminds you to eyeball it
     against the old weights before replacing anything in production.

Usage:
    python retrain.py \
        --batch-dir active_learning_batch1 \
        --weights best.pt \
        --classes "cigarette,plastic_bottle,wrapper,can,other" \
        --output retrain_run1
"""

import argparse
import os
import random
import shutil
import sys

import yaml


def parse_args():
    import datetime
    _today = datetime.date.today().strftime("%Y_%m_%d")
    p = argparse.ArgumentParser(description="Fine-tune YOLO on a reviewed active-learning batch")
    p.add_argument("--batch-dir",
                   default=os.path.join("reviewed_output", f"batch_{_today}"),
                   help="Output folder from review_app.py (contains images/ and labels/) "
                        f"(default: reviewed_output/batch_{_today})")
    p.add_argument("--weights", default="weights/best.pt",
                   help="Existing .pt to fine-tune from (default: weights/best.pt)")
    p.add_argument("--classes", default="litter",
                   help="Comma-separated class names, SAME ORDER as review_app.py used "
                        "(default: litter)")
    p.add_argument("--output",
                   default=os.path.join("retrain_runs", f"run_{_today}"),
                   help="Folder for the built dataset + training run "
                        f"(default: retrain_runs/run_{_today})")
    p.add_argument("--val-split", type=float, default=0.2, help="Fraction held out for validation")
    p.add_argument("--epochs", type=int, default=15, help="Keep this low for correction batches")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.001, help="Low LR so we nudge, not overwrite")
    p.add_argument("--freeze", type=int, default=10,
                   help="Freeze this many backbone layers (0 to disable). Helps prevent "
                        "forgetting when fine-tuning on a small/narrow batch.")
    p.add_argument("--device", default=None,
                   help="'cpu', '0' for first CUDA GPU, or leave unset for auto-detect")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_dataset(batch_dir, output_dir, class_names, val_split, seed):
    images_dir = os.path.join(batch_dir, "images")
    labels_dir = os.path.join(batch_dir, "labels")
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        sys.exit(f"Expected {images_dir} and {labels_dir} to exist (output of review_app.py)")

    files = sorted(f for f in os.listdir(images_dir)
                    if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not files:
        sys.exit(f"No images found in {images_dir}")

    random.Random(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_split))
    val_files = set(files[:n_val])
    train_files = [f for f in files if f not in val_files]

    dataset_dir = os.path.join(output_dir, "dataset")
    for split, split_files in (("train", train_files), ("val", sorted(val_files))):
        img_out = os.path.join(dataset_dir, "images", split)
        lbl_out = os.path.join(dataset_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for f in split_files:
            stem = os.path.splitext(f)[0]
            shutil.copy2(os.path.join(images_dir, f), os.path.join(img_out, f))
            label_src = os.path.join(labels_dir, stem + ".txt")
            label_dst = os.path.join(lbl_out, stem + ".txt")
            if os.path.isfile(label_src):
                shutil.copy2(label_src, label_dst)
            else:
                # Confirmed-negative frame: empty label file is valid YOLO input.
                open(label_dst, "w").close()

    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    data_yaml = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"Dataset built: {len(train_files)} train / {len(val_files)} val -> {dataset_dir}")
    return data_yaml_path


def main():
    args = parse_args()
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not class_names:
        sys.exit("--classes must list at least one class name")
    if not os.path.isfile(args.weights):
        sys.exit(f"Weights not found: {args.weights}")

    data_yaml_path = build_dataset(args.batch_dir, args.output, class_names, args.val_split, args.seed)

    from ultralytics import YOLO
    print(f"Loading base weights: {args.weights}")
    model = YOLO(args.weights)

    train_kwargs = dict(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        lr0=args.lr0,
        project=os.path.abspath(args.output),  # absolute path prevents YOLO nesting under runs/detect/
        name="finetune",
        exist_ok=True,
    )
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze
    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"Fine-tuning: epochs={args.epochs} lr0={args.lr0} freeze={args.freeze}")
    results = model.train(**train_kwargs)

    run_dir = os.path.join(os.path.abspath(args.output), "finetune")
    new_weights = os.path.join(run_dir, "weights", "best.pt")
    print("\n" + "=" * 60)
    print(f"Done. New weights: {new_weights}")
    print("Do NOT overwrite your production weights yet.")
    print("Next: run both old and new weights on the same held-out")
    print("images/video and compare before promoting the new one.")
    print("=" * 60)


if __name__ == "__main__":
    main()
