import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from config_loader import (
    YOLO_DATASET_YAML,
    MODEL_PATH as CONFIG_MODEL_PATH,
    MERGED_DATASET_ROOT,
)

print("=== STARTING ACTIVE LEARNING PIPELINE ===")

# === CLI ARGS ===
parser = argparse.ArgumentParser()
parser.add_argument(
    "--clean", action="store_true", help="Clean dataset folders before pipeline"
)
args = parser.parse_args()

# === CLEAN FOLDERS IF NEEDED ===
if args.clean:
    print("🧹 Cleaning dataset folders before starting pipeline...")
    subprocess.run([sys.executable, "cleanup_dataset_folders.py"])
else:
    print("⚠️ Skipping dataset cleanup (default behavior, no --clean flag)")

# === FOLDER PATHS ===
merged_images = MERGED_DATASET_ROOT / "images/train"
merged_labels = MERGED_DATASET_ROOT / "labels/train"
train_images = list(merged_images.glob("*"))
train_labels = list(merged_labels.glob("*.txt"))


# === STEP 1: Select model ===
def get_latest_model_path(base_dir="runs/obb"):
    base_dir = Path(base_dir)
    run_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        best = run_dir / "weights/best.pt"
        if best.exists():
            return best
    raise FileNotFoundError("❌ No valid best.pt found in any run folder.")


if CONFIG_MODEL_PATH.exists():
    MODEL_PATH = CONFIG_MODEL_PATH
    print(f"📌 MODEL USED: {MODEL_PATH}")
else:
    print(f"⚠️ Configured MODEL_PATH not found: {CONFIG_MODEL_PATH}")
    MODEL_PATH = get_latest_model_path()
    print(f"📌 Falling back to: {MODEL_PATH}")

# === STEP 2: Launch Review Tool ===
print("🔍 Launching manual_review.py...")
try:
    subprocess.run([sys.executable, "manual_review.py"], check=True)
except Exception as e:
    print(f"❌ manual_review.py failed: {e}")
    sys.exit(1)

# === STEP 3: Merge Labels ===
print("🧪 Running boost_merge_labels.py...")
if subprocess.run([sys.executable, "boost_merge_labels.py"]).returncode != 0:
    print("❌ boost_merge_labels.py failed")
    sys.exit(1)

# === STEP 4: Backup old run ===
TRAIN_DIR = Path("runs/obb/train")
BACKUP_DIR = Path("runs/obb/previous-train")

if TRAIN_DIR.exists():
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
        print("🧹 Removed old previous-train folder")
    shutil.move(str(TRAIN_DIR), str(BACKUP_DIR))
    print("🔄 Renamed train → previous-train")

# === STEP 5: Train Model ===
# Delete any leftover .bak files from label corrections
for txt_file in merged_labels.glob("*.bak"):
    txt_file.unlink()
    print(f"🗑️ Deleted leftover backup file: {txt_file.name}")

if not train_images or not train_labels:
    print("❌ No training data found. Skipping training.")
else:
    print(f"✅ Found {len(train_images)} images and {len(train_labels)} labels.")
    train_args = [
        "yolo",
        "task=obb",
        "mode=train",
        f"model={str(MODEL_PATH)}",
        f"data={YOLO_DATASET_YAML}",
        "imgsz=960",
        "device=0",
        "name=train",
        "resume=False",
        "val=False",
        "epochs=20",
    ]
    print("🚀 Running YOLO training...")
    result = subprocess.run(train_args)
    if result.returncode != 0:
        print("❌ YOLO training failed to execute properly.")
        sys.exit(1)

    # === STEP 6: Update model if needed ===
    final_best = Path("runs/obb/train/weights/best.pt")
    target_model = CONFIG_MODEL_PATH
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_model = Path(f"temp/last_model_{timestamp}.pt")

    if final_best.exists():
        backup_model.parent.mkdir(parents=True, exist_ok=True)
        if target_model.exists() and final_best.resolve() != target_model.resolve():
            shutil.copy2(target_model, backup_model)
            print(f"📦 Backed up old model to: {backup_model}")
        if final_best.resolve() != target_model.resolve():
            shutil.copy2(final_best, target_model)
            print(f"✅ Updated MODEL_PATH with new best.pt: {target_model}")
        else:
            print("⚠️ Skipping model copy — already latest.")
    else:
        print("❌ Training finished, but no best.pt found at expected location.")
        print("🧹 Cleaning up broken run folder...")
        shutil.rmtree("runs/obb/train", ignore_errors=True)

# === STEP 7: Evaluate Model ===
eval_dir = Path("eval_output")
shutil.rmtree(eval_dir / "post_active_learning", ignore_errors=True)
eval_dir.mkdir(parents=True, exist_ok=True)

if CONFIG_MODEL_PATH.exists():
    print(f"📊 Evaluating {len(train_images)} images using updated model...")

    eval_args = [
        "yolo",
        "task=obb",
        "mode=predict",
        f"model={CONFIG_MODEL_PATH}",
        f"source={merged_images}",
        "imgsz=960",
        "conf=0.25",
        "iou=0.5",
        "device=0",
        "show=False",
        "save=True",
        "save_txt=False",
        "project=eval_output",
        "name=post_active_learning",
        "exist_ok=True",
    ]

    subprocess.run(eval_args)
else:
    print(f"⚠️ Skipping evaluation — model not found at {CONFIG_MODEL_PATH}")
