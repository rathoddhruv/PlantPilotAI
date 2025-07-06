import os
import cv2
import shutil
from ultralytics import YOLO
from pathlib import Path
from tabulate import tabulate

# === CONFIG ===
model_path = r"runs\detect\plant_detector_boosted\weights\best.pt"
image_folder = r"C:\AI\PlantPilotAI\data\yolo_dataset\images\train"
active_label_dir = Path("active_labels")
manual_review_dir = Path("active_review")
imgsz = 960
uncertain_threshold = 0.35

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
    results = model.predict(source=str(img_path), imgsz=960, conf=0.05)
    result = results[0]
    img = cv2.imread(str(img_path))
    names = result.names

    for cls_id, conf, xyxy in zip(
        result.boxes.cls, result.boxes.conf, result.boxes.xyxy
    ):
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names[int(cls_id)]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,  # font size
            (0, 255, 0),
            3,
        )

    cv2.imshow(f"Prediction - {img_path.name}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    result = results[0]
    row = [img_path.name]

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
                print(f"❓ Uncertain: {label} ({conf_pct}%) - label this? (y/n): ", end="")
                choice = input().strip().lower()
                if choice == "y":
                    active_label_dir.mkdir(exist_ok=True)
                    label_file = active_label_dir / f"{img_path.stem}.txt"
                    box = boxes.xywhn[i].tolist()  # normalized x_center, y_center, w, h
                    with open(label_file, "w") as f:
                        f.write(f"{int(cls_id)} {' '.join(map(str, box))}\n")
                    print(f"✅ Saved manual label to {label_file}")
            else:
                print(f"✅ Detected: {label} ({conf_pct}%)")
                detected_labels.append(f"{label} ({conf_pct}%)")

        row.append(", ".join(detected_labels) if detected_labels else "Uncertain")

        # Show result image
        result_img_path = Path(result.save_dir) / img_path.name
        img = cv2.imread(str(result_img_path))
        if img is not None:
            cv2.imshow(f"Prediction - {img_path.name}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"⚠️ Could not load result image: {result_img_path}")
    else:
        print("⚠️ No detections. Label manually? (y/n): ", end="")
        row.append("None")
        choice = input().strip().lower()
        if choice == "y":
            manual_review_dir.mkdir(exist_ok=True)
            shutil.copy(str(img_path), manual_review_dir / img_path.name)
            print(f"✅ Copied image to {manual_review_dir}")

    summary.append(row)

# === Summary Table ===
print("\n📊 Detection Summary:")
print(tabulate(summary, headers=["Image", "Detections"], tablefmt="grid"))
