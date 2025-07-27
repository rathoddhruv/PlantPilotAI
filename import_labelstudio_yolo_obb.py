import zipfile
from pathlib import Path
import shutil

EXPORTS_DIR = Path("label_studio_exports")
YOLO_DATASET_ROOT = Path("data/yolo_dataset")
DEST_IMAGES = YOLO_DATASET_ROOT / "images/train"
DEST_LABELS = YOLO_DATASET_ROOT / "labels/train"
DEST_META = YOLO_DATASET_ROOT
TEMP_UNZIP_DIR = YOLO_DATASET_ROOT / "temp/ls_extract"

# Step 1: Find latest zip
zip_files = sorted(EXPORTS_DIR.glob("*.zip"), key=lambda z: z.stat().st_mtime, reverse=True)
if not zip_files:
    print("❌ No ZIP export found in label_studio_exports/")
    exit(1)

latest_zip = zip_files[0]
print(f"🗜️ Found ZIP export: {latest_zip.name}")

# Step 2: Extract zip
if TEMP_UNZIP_DIR.exists():
    shutil.rmtree(TEMP_UNZIP_DIR)
TEMP_UNZIP_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(latest_zip, 'r') as zip_ref:
    zip_ref.extractall(TEMP_UNZIP_DIR)

# Step 3: Search for images and labels folders
found_images = list(TEMP_UNZIP_DIR.rglob("images"))
found_labels = list(TEMP_UNZIP_DIR.rglob("labels"))

if not found_images or not found_labels:
    print("❌ Could not find both 'images/' and 'labels/' folders inside the ZIP.")
    exit(1)

src_images = found_images[0]
src_labels = found_labels[0]
print(f"📁 Found image folder: {src_images}")
print(f"📁 Found label folder: {src_labels}")

# Step 4: Move files
DEST_IMAGES.mkdir(parents=True, exist_ok=True)
DEST_LABELS.mkdir(parents=True, exist_ok=True)

for file in src_images.glob("*"):
    if file.is_file():
        shutil.move(str(file), DEST_IMAGES / file.name)

for file in src_labels.glob("*.txt"):
    with open(file, "r") as f:
        lines = f.readlines()
    # Keep only lines with 5 values (YOLO box format)
    valid_lines = [line for line in lines if len(line.strip().split()) == 5]
    if valid_lines:
        with open(DEST_LABELS / file.name, "w") as out:
            out.writelines(valid_lines)

# Step 5: Optional meta files
for extra in ["classes.txt", "notes.json"]:
    match = list(TEMP_UNZIP_DIR.rglob(extra))
    if match:
        shutil.move(str(match[0]), DEST_META / match[0].name)
        print(f"📄 Copied {extra}")

print("✅ YOLO dataset import completed successfully!")
