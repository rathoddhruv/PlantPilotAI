import os
import shutil
from pathlib import Path

# source folder with subfolders
src_root = Path(r"C:\Data\Projects\PlantPilotAI\images")
# destination flattened folder
dst_root = Path(r"C:\Data\Projects\PlantPilotAI\flattened_images")
dst_root.mkdir(exist_ok=True)

image_extensions = [".jpg", ".jpeg", ".png"]
counter = 0

for class_folder in src_root.iterdir():
    if class_folder.is_dir():
        class_name = class_folder.name
        for img_path in class_folder.glob("*"):
            if img_path.suffix.lower() in image_extensions:
                new_filename = f"{class_name}_{img_path.name}"
                dst_path = dst_root / new_filename
                shutil.copy2(img_path, dst_path)
                counter += 1

print(f"✅ Flattened {counter} images into {dst_root}")
