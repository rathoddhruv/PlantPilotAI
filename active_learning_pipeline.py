# File: active_learning_pipeline.py
import subprocess
import os
from pathlib import Path
import sys
from config_loader import YOLO_DATASET_YAML, MODEL_PATH

from config_loader import MERGED_DATASET_ROOT
merged_images = MERGED_DATASET_ROOT / "images/train"


print("≡ƒöü STARTING ACTIVE LEARNING PIPELINE...")

# ≡ƒƒí STEP 1: interactive labeling
print("≡ƒƒí Launching manual_review.py...")
try:
    with open("manual_review.py", encoding="utf-8") as f:
        exec(f.read())
except Exception as e:
    print(f"Γ¥î manual_review.py failed: {e}")
    exit(1)

# ≡ƒƒí STEP 2: merge reviewed labels
print("≡ƒƒí Running boost_merge_labels.py...")
merge_result = subprocess.run([sys.executable, "boost_merge_labels.py"])
if merge_result.returncode != 0:
    print("Γ¥î boost_merge_labels.py failed")
    exit(1)

# ≡ƒƒí STEP 3: decide resume or not
print("≡ƒƒó Training from base model (no resume)")


# ≡ƒƒí STEP 4: train the model (always reuse MODEL_PATH)
# skip resume logic
train_args = [
    "yolo",
    "task=obb",
    "mode=train",
    f"data={YOLO_DATASET_YAML}",
    f"imgsz=960",
    f"device=0",
    f"name=train",
    f"project={MODEL_PATH.parents[2]}",  # runs/obb
    f"model={MODEL_PATH}",  # always start fresh from current model
]

print("≡ƒƒó Running training...")
subprocess.run(train_args)

# === STEP 5: Replace original model with latest best.pt
from shutil import copy2
from datetime import datetime

final_best = Path("runs/obb/train/weights/best.pt")
target_model = MODEL_PATH  # from config_loader.py

# versioned backup filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_model = Path(f"temp/last_model_{timestamp}.pt")

if final_best.exists():
    backup_model.parent.mkdir(exist_ok=True)
    if target_model.exists() and target_model.resolve() != final_best.resolve():
        copy2(target_model, backup_model)
        print(f"🔁 Backed up old model to: {backup_model}")
        copy2(final_best, target_model)
        print(f"✅ Updated MODEL_PATH with new best.pt: {target_model}")
    else:
        print("⚠️ MODEL_PATH already up-to-date (same file), no copy needed.")
else:
    print("⚠️ Training completed, but no best.pt found to update.")


# === STEP 6: Validate updated model on reviewed images ===
from ultralytics import YOLO
from config_loader import MERGED_DATASET_ROOT

merged_images = MERGED_DATASET_ROOT / "images/train"
eval_dir = Path("eval_output")
eval_dir.mkdir(exist_ok=True)

model = YOLO(str(MODEL_PATH))
image_paths = list(merged_images.glob("*"))

print(f"🖼️ Evaluating {len(image_paths)} images with updated model...")

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

