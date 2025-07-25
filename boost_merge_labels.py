import shutil
from pathlib import Path
import yaml
from config_loader import (
    CLASS_MAP_REVERSE,
    ORIGINAL_IMAGES,
    ORIGINAL_LABELS,
    ACTIVE_LABEL_DIR,
    TEST_IMAGE_FOLDER,
    MERGED_DATASET_ROOT,
    YOLO_DATASET_YAML,
)


# === CLEAN YOLO_MERGED FOLDERS ===
def clean_yolo_merged():
    print("🧹 cleaning data/yolo_merged/images/train and labels/train...")
    for sub in ["images/train", "labels/train"]:
        path = Path("data/yolo_merged") / sub
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


clean_yolo_merged()

# === CONFIG ===
merged_root = MERGED_DATASET_ROOT
merged_images = merged_root / "images/train"
merged_labels = merged_root / "labels/train"

# === PREP FOLDERS ===
for path in [merged_images, merged_labels]:
    path.mkdir(parents=True, exist_ok=True)

# === COPY ORIGINAL IMAGES + LABELS ===
image_files = list(ORIGINAL_IMAGES.glob("*"))
for img_file in image_files:
    label_file = ORIGINAL_LABELS / f"{img_file.stem}.txt"
    shutil.copy(img_file, merged_images / img_file.name)
    if label_file.exists():
        shutil.copy(label_file, merged_labels / label_file.name)

# === COPY ACTIVE LABELS + MATCHED IMAGES ===
active_files = list(ACTIVE_LABEL_DIR.glob("*.txt"))
copied_images = 0

for label_path in active_files:
    shutil.copy(label_path, merged_labels / label_path.name)

    # check for matching image in test folder
    for ext in [".jpg", ".jpeg", ".png"]:
        image_path = TEST_IMAGE_FOLDER / f"{label_path.stem}{ext}"
        if image_path.exists():
            shutil.copy(image_path, merged_images / image_path.name)
            image_path.unlink()  # delete from test folder
            copied_images += 1
            break

    # remove old active label file
    label_path.unlink()

print(
    f"✅ {len(image_files)} original images + {len(active_files)} active labels copied"
)
print(f"✅ {copied_images} new images copied and cleaned from test folder")
print("🧹 cleaned up used active labels and test images")

# === GENERATE YOLO DATASET YAML ===
dataset_yaml = {
    "path": str(merged_root),
    "train": "images/train",
    "val": "images/train",  # still required by YOLO CLI, even if val == train
    "names": {idx: name for idx, name in CLASS_MAP_REVERSE.items()},
}
merged_images_dir = Path("data/yolo_merged/images/train")
if not any(merged_images_dir.glob("*")):
    print("❌ No merged training images found. Exiting.")
    exit(1)


with open(YOLO_DATASET_YAML, "w") as f:
    yaml.dump(dataset_yaml, f, sort_keys=False)

print(f"✅ yolo_dataset.yaml updated at {YOLO_DATASET_YAML}")
print("🎯 dataset ready at:", merged_root)


# === FIX CORRUPT LABELS ===
from pathlib import Path

LABEL_FOLDER = MERGED_DATASET_ROOT / "labels/train"
for label_file in LABEL_FOLDER.glob("*.txt"):
    lines = label_file.read_text().strip().splitlines()
    cleaned = []
    corrupted = False

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            corrupted = True
            continue
        try:
            floats = [float(x) for x in parts]
            cleaned.append(" ".join(map(str, floats)))
        except ValueError:
            corrupted = True

    if corrupted:
        backup = label_file.with_suffix(".bak")
        if backup.exists():
            backup.unlink()
        label_file.rename(backup)
        label_file.write_text("\n".join(cleaned))
        print(f"🔁 Fixed: {label_file.name}, backup saved as {backup.name}")
