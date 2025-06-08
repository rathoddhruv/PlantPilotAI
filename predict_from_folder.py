import os
from ultralytics import YOLO
import cv2
from pathlib import Path
from tabulate import tabulate

# === CONFIG ===
model_path = "runs/detect/train/weights/best.pt"
image_folder = r"C:\AI\PlantPilotAI\data\yolo_dataset\images\train"
imgsz = 640

# === Load model ===
model = YOLO(model_path)
image_paths = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.jpeg")) + list(Path(image_folder).glob("*.png"))

if not image_paths:
    print("❌ No images found in the folder!")
    exit()

# === Prediction results tracker ===
summary = []

# === Predict on all images ===
for img_path in image_paths:
    print(f"\n🔍 Predicting on: {img_path.name}")
    results = model.predict(source=str(img_path), imgsz=imgsz, conf=0.05, save=True)

    result = results[0]
    row = [img_path.name]

    if result.boxes and result.boxes.cls.numel() > 0:
        names = result.names
        boxes = result.boxes
        classes = boxes.cls.tolist()
        scores = boxes.conf.tolist()

        detected_labels = []
        for i, cls_id in enumerate(classes):
            label = names[int(cls_id)]
            conf = round(scores[i] * 100, 1)
            print(f"✅ Detected: {label} ({conf}%)")
            detected_labels.append(f"{label} ({conf}%)")

        row.append(", ".join(detected_labels))

        # show result image
        result_img_path = Path(result.save_dir) / img_path.name
        img = cv2.imread(str(result_img_path))
        if img is not None:
            cv2.imshow(f"Prediction - {img_path.name}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"⚠️ Could not load result image: {result_img_path}")
    else:
        print("⚠️ No detections.")
        row.append("None")

    summary.append(row)

# === Final summary table ===
print("\n📊 Detection Summary:")
print(tabulate(summary, headers=["Image", "Detections"], tablefmt="grid"))
