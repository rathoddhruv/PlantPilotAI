import json
import shutil
from pathlib import Path
import re

json_file = Path("project-8-at-2025-05-24-09-46-1bdeaf3b.json")
flattened_images = Path("flattened_images")
output_base = Path("data/yolo_dataset")
images_output = output_base / "images" / "train"
labels_output = output_base / "labels" / "train"

images_output.mkdir(parents=True, exist_ok=True)
labels_output.mkdir(parents=True, exist_ok=True)

with open(json_file, "r", encoding="utf-8") as f:
    label_data = json.load(f)

class_map = {
    "Dandelions": 0,
    "Hydrangea": 1,
    "Dandelions": 2
}


converted, skipped = 0, 0
for task in label_data:
    # remove prefix like UUID- from filename
    image_path = task["data"]["image"].split("/")[-1]
    image_path = "-".join(image_path.split("-")[1:])  # remove first part (uuid)

    image_path = image_path.replace("%20", " ")  # if exported with spaces
    image_path = image_path.replace("(", "_").replace(")", "_")
    src_img = flattened_images / image_path
    if not src_img.exists():
        skipped += 1
        continue

    shutil.copy2(src_img, images_output / src_img.name)

    annotations = task["annotations"][0]["result"]
    yolo_lines = []
    for result in annotations:
        if result["type"] != "rectanglelabels":
            continue
        label = result["value"]["rectanglelabels"][0]
        if label not in class_map:
            continue
        class_id = class_map[label]

        bbox = result["value"]
        orig_w = result["original_width"]
        orig_h = result["original_height"]

        x = bbox["x"] / 100 * orig_w
        y = bbox["y"] / 100 * orig_h
        w = bbox["width"] / 100 * orig_w
        h = bbox["height"] / 100 * orig_h

        xc = (x + w / 2) / orig_w
        yc = (y + h / 2) / orig_h
        nw = w / orig_w
        nh = h / orig_h

        yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

    label_path = labels_output / f"{src_img.stem}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    converted += 1

print(f"✅ Converted: {converted} images")
print(f"⚠️ Skipped: {skipped} images (not found)")
