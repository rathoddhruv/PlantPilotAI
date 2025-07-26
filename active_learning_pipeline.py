# File: active_learning_pipeline.py
import subprocess
import os
from pathlib import Path
import sys
from config_loader import YOLO_DATASET_YAML, MODEL_PATH
from config_loader import MERGED_DATASET_ROOT
from shutil import copy2
from datetime import datetime
from ultralytics import YOLO
import shutil

merged_images = MERGED_DATASET_ROOT / "images/train"

print("=== STARTING ACTIVE LEARNING PIPELINE ===")

# CLEAN ORIGINAL & MERGED DATASET FOLDERS BEFORE START
print("🧹 Cleaning dataset folders before starting pipeline...")
subprocess.run([sys.executable, "cleanup_dataset_folders.py"])

# STEP 1: Interactive labeling
print("Launching manual_review.py...")
try:
    subprocess.run([sys.executable, "manual_review.py"])
except Exception as e:
    print(f"manual_review.py failed: {e}")
    exit(1)

# STEP 2: Merge reviewed labels
print("Running boost_merge_labels.py...")
merge_result = subprocess.run([sys.executable, "boost_merge_labels.py"])
if merge_result.returncode != 0:
    print("boost_merge_labels.py failed")
    exit(1)

# STEP 3: Always train from base model (no resume)
print("Training from base model (no resume)")

# STEP 4: Train the model
train_args = [
    "yolo",
    "task=obb",
    "mode=train",
    f"data={YOLO_DATASET_YAML}",
    "imgsz=960",
    "device=0",
    "name=train",
    f"project={MODEL_PATH.parents[2]}",
    f"model={MODEL_PATH}",
    "val=False",
]

print("Running training...")
subprocess.run(train_args)

# STEP 5: Replace original model with latest best.pt
final_best = Path("runs/obb/train/weights/best.pt")
target_model = MODEL_PATH
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_model = Path(f"temp/last_model_{timestamp}.pt")

if final_best.exists():
    backup_model.parent.mkdir(exist_ok=True)
    if target_model.exists() and target_model.resolve() != final_best.resolve():
        copy2(target_model, backup_model)
        print(f"Backed up old model to: {backup_model}")
        copy2(final_best, target_model)
        print(f"Updated MODEL_PATH with new best.pt: {target_model}")
    else:
        print("MODEL_PATH already up-to-date, no copy needed.")
else:
    print("Training completed, but no best.pt found to update.")

# STEP 6: Validate updated model on reviewed images
eval_dir = Path("eval_output")
eval_dir.mkdir(exist_ok=True)
model = YOLO(str(MODEL_PATH))
image_paths = list(merged_images.glob("*"))

print(f"Evaluating {len(image_paths)} images with updated model...")
shutil.rmtree(eval_dir / "post_active_learning", ignore_errors=True)

model.predict(
    source=[str(p) for p in image_paths],
    save=True,
    project=str(eval_dir),
    name="post_active_learning",
    imgsz=960,
    conf=0.25,
    iou=0.5,
    device=0,
    show=False,
    save_txt=False
)
