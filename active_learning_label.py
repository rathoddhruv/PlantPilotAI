import os
import cv2
import shutil
from ultralytics import YOLO
from pathlib import Path
from tabulate import tabulate
import subprocess
from config_loader import CLASS_NAMES, CLASS_MAP, MODEL_PATH, TEST_IMAGE_FOLDER, ACDSEE_PATH, ACTIVE_LABEL_DIR, MANUAL_REVIEW_DIR, WRONG_LABEL_DIR, SAVE_DIR, UNCERTAIN_THRESHOLD, MERGED_DATASET_ROOT

# === CONFIG ===
model_path = MODEL_PATH
image_folder = TEST_IMAGE_FOLDER
active_label_dir = ACTIVE_LABEL_DIR
manual_review_dir = MANUAL_REVIEW_DIR
wrong_label_dir = WRONG_LABEL_DIR
save_dir = SAVE_DIR
imgsz = 960
uncertain_threshold = UNCERTAIN_THRESHOLD
acdsee_path = ACDSEE_PATH

# === Prepare folders ===
save_dir.mkdir(exist_ok=True, parents=True)
active_label_dir.mkdir(exist_ok=True)
manual_review_dir.mkdir(exist_ok=True)
wrong_label_dir.mkdir(exist_ok=True)

# === Load model ===
model = YOLO(model_path)
image_paths = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.jpeg")) + list(Path(image_folder).glob("*.png"))

if not image_paths:
    print("❌ No images found in the folder!")
    exit()

print("📌 Select review mode:")
print("1. Review all images")
print(f"2. Review only detections with confidence < {int(uncertain_threshold * 100)}%")
mode_choice = input("Enter 1 or 2: ").strip()
review_all = mode_choice == "1"

summary = []

# === Active Learning Prediction Loop ===
for img_path in image_paths:
    print(f"\n🔍 Predicting on: {img_path.name}")
    results = model.predict(source=str(img_path), imgsz=imgsz, conf=0.05, save=True, save_dir=str(save_dir), line_thickness=3)

    result = results[0]
    row = [img_path.name]
    already_shown = False

    if result.boxes and result.boxes.cls.numel() > 0:
        names = result.names
        boxes = result.boxes
        classes = boxes.cls.tolist()
        scores = boxes.conf.tolist()
        xywhn = boxes.xywhn.tolist()

        detections = sorted(zip(classes, scores, xywhn), key=lambda x: x[1])

        detected_labels = []
        skip_image = False
        skip_as_correct = False
        skip_all_others_wrong = False
        mark_rest_as_correct = False
        mark_rest_as_wrong = False

        for idx, (cls_id, conf, box) in enumerate(detections):
            label = names[int(cls_id)]
            conf_pct = round(conf * 100, 1)

            if not review_all and conf >= uncertain_threshold:
                detected_labels.append(f"{label} ({conf_pct}%)")
                continue

            result_img_path = Path(result.save_dir) / img_path.name
            if not already_shown and result_img_path.exists():
                subprocess.Popen([acdsee_path, str(result_img_path)], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                already_shown = True

            print(f"❓ Review: {label} ({conf_pct}%) - (y = correct, w = wrong, s = all correct, c = skip, n = rest wrong, a = rest correct, b = rest wrong): ", end="")
            choice = input().strip().lower()

            if choice == "s":
                skip_as_correct = True
                break
            elif choice == "c":
                skip_image = True
                break
            elif choice == "n":
                skip_all_others_wrong = True
                break
            elif choice == "a":
                mark_rest_as_correct = True
                break
            elif choice == "b":
                mark_rest_as_wrong = True
                break
            elif choice == "y":
                label_file = active_label_dir / f"{img_path.stem}.txt"
                with open(label_file, "a") as f:
                    f.write(f"{int(cls_id)} {' '.join(map(str, box))}\n")
                print(f"✅ Saved to active_labels: {label_file}")
                shutil.copy(str(img_path), MERGED_DATASET_ROOT / "images/train")
            elif choice == "w":
                label_file = wrong_label_dir / f"{img_path.stem}.txt"
                with open(label_file, "a") as f:
                    f.write(f"{int(cls_id)} {' '.join(map(str, box))}\n")
                print(f"❌ Marked as incorrect in: {label_file}")

        if skip_as_correct:
            label_file = active_label_dir / f"{img_path.stem}.txt"
            for cls_id, conf, box in detections:
                with open(label_file, "a") as f:
                    f.write(f"{int(cls_id)} {' '.join(map(str, box))}\n")
            print(f"✅ All detections saved as correct for: {img_path.name}")
            shutil.copy(str(img_path), MERGED_DATASET_ROOT / "images/train")

        elif skip_all_others_wrong:
            label_file = wrong_label_dir / f"{img_path.stem}.txt"
            for cls_id, conf, box in detections[idx:]:
                with open(label_file, "a") as f:
                    f.write(f"{int(cls_id)} {' '.join(map(str, box))}\n")
            print(f"❌ Remaining detections marked as wrong for: {img_path.name}")

        elif mark_rest_as_correct:
            label_file = active_label_dir / f"{img_path.stem}.txt"
            for cls_id, conf, box in detections[idx:]:
                with open(label_file, "a") as f:
                    f.write(f"{int(cls_id)} {' '.join(map(str, box))}\n")
            print(f"✅ Remaining detections marked as correct for: {img_path.name}")

        elif mark_rest_as_wrong:
            label_file = wrong_label_dir / f"{img_path.stem}.txt"
            for cls_id, conf, box in detections[idx:]:
                with open(label_file, "a") as f:
                    f.write(f"{int(cls_id)} {' '.join(map(str, box))}\n")
            print(f"❌ Remaining detections marked as wrong for: {img_path.name}")

        row.append(", ".join(detected_labels) if detected_labels else "Uncertain")

    else:
        print("⚠️ No detections. Label manually? (y/n): ", end="")
        row.append("None")
        choice = input().strip().lower()
        if choice == "y":
            shutil.copy(str(img_path), manual_review_dir / img_path.name)
            print(f"✅ Copied image to {manual_review_dir}")

    summary.append(row)
    cv2.destroyAllWindows()

# === Summary Table ===
print("\n📊 Detection Summary:")
print(tabulate(summary, headers=["Image", "Detections"], tablefmt="grid"))