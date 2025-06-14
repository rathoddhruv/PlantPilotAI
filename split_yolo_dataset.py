import os
import shutil
import random
from pathlib import Path

# config
base_dir = Path("data/yolo_dataset")
img_dir = base_dir / "images"
lbl_dir = base_dir / "labels"

output_img_train = img_dir / "train"
output_img_val = img_dir / "val"
output_lbl_train = lbl_dir / "train"
output_lbl_val = lbl_dir / "val"

# check for missing label files
all_images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
missing = [img.name for img in all_images if not (lbl_dir / f"{img.stem}.txt").exists()]

print(f"{len(missing)} images have no matching labels")
if missing:
    print("Missing examples:", missing[:5])

# make dirs
for d in [output_img_train, output_img_val, output_lbl_train, output_lbl_val]:
    d.mkdir(parents=True, exist_ok=True)

# filter only valid images that have label files
valid_images = [img for img in all_images if (lbl_dir / f"{img.stem}.txt").exists()]
random.shuffle(valid_images)

# split ratio
split_ratio = 0.8
split_index = int(len(valid_images) * split_ratio)
train_imgs = valid_images[:split_index]
val_imgs = valid_images[split_index:]

# move image-label pairs
def move_pair(img_path, dest_img, dest_lbl):
    label_path = lbl_dir / f"{img_path.stem}.txt"
    shutil.move(str(img_path), dest_img / img_path.name)
    shutil.move(str(label_path), dest_lbl / label_path.name)

for img in train_imgs:
    move_pair(img, output_img_train, output_lbl_train)

for img in val_imgs:
    move_pair(img, output_img_val, output_lbl_val)

print(f"✅ Split complete! {len(train_imgs)} train images, {len(val_imgs)} val images.")
