import shutil
from pathlib import Path
import random

# === CONFIG ===
original_images = Path("data/yolo_dataset/images/train")
original_labels = Path("data/yolo_dataset/labels/train")
active_labels = Path("active_labels")

merged_root = Path("data/yolo_merged")
merged_train_images = merged_root / "images/train"
merged_val_images = merged_root / "images/val"
merged_train_labels = merged_root / "labels/train"
merged_val_labels = merged_root / "labels/val"

# === PREP FOLDERS ===
for path in [merged_train_images, merged_val_images, merged_train_labels, ]:
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

# === OVERWRITE WITH ACTIVE LABELS IN TRAIN SPLIT ONLY ===
active_files = list(active_labels.glob("*.txt"))
for f in active_files:
    shutil.copy(f, merged_train_labels / f.name)

print(f"✅ {len(train_files)} train + {len(val_files)} val images ready with active labels injected")
print("🎯 dataset folder: data/yolo_merged")
