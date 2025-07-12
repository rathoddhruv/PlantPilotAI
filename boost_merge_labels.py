import shutil
from pathlib import Path
import random
import yaml
from config_loader import (
    CLASS_NAMES,
    CLASS_MAP,
    CLASS_MAP_REVERSE,
    ORIGINAL_IMAGES,
    ORIGINAL_LABELS,
    ACTIVE_LABEL_DIR,
    TEST_IMAGE_FOLDER,
    MERGED_DATASET_ROOT,
    YOLO_DATASET_YAML
)

# === CONFIG ===
original_images = ORIGINAL_IMAGES
original_labels = ORIGINAL_LABELS
active_labels = ACTIVE_LABEL_DIR
test1_images = TEST_IMAGE_FOLDER

merged_root = MERGED_DATASET_ROOT
merged_train_images = merged_root / "images/train"
merged_val_images = merged_root / "images/val"
merged_train_labels = merged_root / "labels/train"
merged_val_labels = merged_root / "labels/val"

# === PREP FOLDERS ===
for path in [
    merged_train_images,
    merged_val_images,
    merged_train_labels,
    merged_val_labels,
]:
    path.mkdir(parents=True, exist_ok=True)

# === GATHER IMAGE FILES ===
image_files = list(original_images.glob("*"))
random.shuffle(image_files)
split_idx = int(len(image_files) * 0.9)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# === COPY TRAIN IMAGES AND LABELS ===
for img_file in train_files:
    label_file = original_labels / f"{img_file.stem}.txt"
    shutil.copy(img_file, merged_train_images / img_file.name)
    if label_file.exists():
        shutil.copy(label_file, merged_train_labels / label_file.name)

# === COPY VAL IMAGES AND LABELS ===
for img_file in val_files:
    label_file = original_labels / f"{img_file.stem}.txt"
    shutil.copy(img_file, merged_val_images / img_file.name)
    if label_file.exists():
        shutil.copy(label_file, merged_val_labels / label_file.name)

# === ADD ACTIVE LABELS + THEIR IMAGES ===
active_files = list(active_labels.glob("*.txt"))
for f in active_files:
    shutil.copy(f, merged_train_labels / f.name)

    # try multiple image formats just in case
    for ext in [".jpg", ".jpeg", ".png"]:
        img_file = test1_images / f.with_suffix(ext).name
        if img_file.exists():
            target_path = merged_train_images / img_file.name
            if not target_path.exists():
                shutil.copy(img_file, target_path)
                print(f"✅ copied {img_file.name}")
            else:
                print(f"⏭️ skipped (already exists): {img_file.name}")
            break

# === GENERATE YOLO DATASET YAML ===
dataset_yaml = {
    "path": str(merged_root),
    "train": "images/train",
    "val": "images/val",
    "names": {idx: name for idx, name in CLASS_MAP_REVERSE.items()}
}

with open(YOLO_DATASET_YAML, "w") as f:
    yaml.dump(dataset_yaml, f, sort_keys=False)

print(f"✅ {len(train_files)} train + {len(val_files)} val images copied")
print(f"✅ {len(active_files)} active labels injected with images from test-1")
print(f"✅ yolo_dataset.yaml generated at {YOLO_DATASET_YAML}")
print("🎯 dataset folder: data/yolo_merged")
