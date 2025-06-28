import os
import cv2
import shutil
from ultralytics import YOLO
from pathlib import Path
from tabulate import tabulate
import subprocess

# === CONFIG ===
model_path = r"runs/detect/plant_detector_boosted/weights/best.pt"
image_folder = r"C:\Data\Projects\PlantPilotAI\data\yolo_dataset\images\train"
active_label_dir = Path("active_labels")
manual_review_dir = Path("active_review")
save_dir = Path("runs/active_review_output")
imgsz = 960
uncertain_threshold = 0.35
acdsee_path = r"C:\Program Files\ACD Systems\ACDSee Pro\6.0\ACDSeePro6.exe"

# === Prepare folders ===
save_dir.mkdir(exist_ok=True, parents=True)
active_label_dir.mkdir(exist_ok=True)
manual_review_dir.mkdir(exist_ok=True)

# === Load model ===
model = YOLO(model_path)
image_paths = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.jpeg")) + list(Path(image_folder).glob("*.png"))

if not image_paths:
    print("❌ No images found in the folder!")
    exit()

summary = []

# === Active Learning Prediction Loop ===
for img_path in image_paths:
    print(f"\n🔍 Predicting on: {img_path.name}")
    results = model.predict(source=str(img_path), imgsz=imgsz, conf=0.05, save=True, save_dir=str(save_dir))
    result = results[0]
    row = [img_path.name]
    already_shown = False

    if result.boxes and result.boxes.cls.numel() > 0:
        names = result.names
        boxes = result.boxes
        classes = boxes.cls.tolist()
        scores = boxes.conf.tolist()

        detected_labels = []
        for i, cls_id in enumerate(classes):
            conf = scores[i]
            label = names[int(cls_id)]
            conf_pct = round(conf * 100, 1)

            if conf < uncertain_threshold:
                # Show uncertain image with OpenCV
                result_img_path = Path(result.save_dir) / img_path.name
                if result_img_path.exists():
                    subprocess.Popen([acdsee_path, str(result_img_path)], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    img = cv2.imread(str(result_img_path))
                    if img is not None:
                        max_width = 900
                        if img.shape[1] > max_width:
                            scale_ratio = max_width / img.shape[1]
                            new_size = (int(img.shape[1] * scale_ratio), int(img.shape[0] * scale_ratio))
                            img = cv2.resize(img, new_size)
                        cv2.imshow(f"Prediction - {img_path.name}", img)
                        cv2.waitKey(1)
                        already_shown = True

                print(f"❓ Uncertain: {label} ({conf_pct}%) - label this? (y/n): ", end="")
                choice = input().strip().lower()
                cv2.destroyAllWindows()

                if choice == "y":
                    label_file = active_label_dir / f"{img_path.stem}.txt"
                    box = boxes.xywhn[i].tolist()
                    with open(label_file, "w") as f:
                        f.write(f"{int(cls_id)} {' '.join(map(str, box))}\n")
                    print(f"✅ Saved manual label to {label_file}")
            else:
                print(f"✅ Detected: {label} ({conf_pct}%)")
                detected_labels.append(f"{label} ({conf_pct}%)")

        row.append(", ".join(detected_labels) if detected_labels else "Uncertain")

        if not already_shown:
            result_img_path = Path(result.save_dir) / img_path.name
            if result_img_path.exists():
                subprocess.Popen([acdsee_path, str(result_img_path)], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print(f"⚠️ Could not load result image: {result_img_path}")
    else:
        print("⚠️ No detections. Label manually? (y/n): ", end="")
        row.append("None")
        choice = input().strip().lower()
        if choice == "y":
            shutil.copy(str(img_path), manual_review_dir / img_path.name)
            print(f"✅ Copied image to {manual_review_dir}")

    summary.append(row)

# === Summary Table ===
print("\n📊 Detection Summary:")
print(tabulate(summary, headers=["Image", "Detections"], tablefmt="grid"))
