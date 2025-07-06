import os
import cv2
import shutil
from ultralytics import YOLO
from pathlib import Path
from tabulate import tabulate
import subprocess

# === CONFIG ===
model_path = r"runs/detect/plant_detector_boosted/weights/best.pt"
image_folder = r"C:\Data\Projects\test-1"
active_label_dir = Path("active_labels")
manual_review_dir = Path("active_review")
wrong_label_dir = Path("wrong_labels")
save_dir = Path("runs/active_review_output")
imgsz = 960
uncertain_threshold = 0.35
acdsee_path = r"C:\Program Files\ACD Systems\ACDSee Pro\6.0\ACDSeePro6.exe"

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
    results = model.predict(
        source=str(img_path),
        imgsz=imgsz,
        conf=0.05,
        save=True,
        save_dir=str(save_dir),
        line_thickness=1,  # thinner boxes
        show_labels=True,
        show_conf=True,
    )

    result = results[0]
    row = [img_path.name]
    already_shown = False

    if result.boxes and result.boxes.cls.numel() > 0:
        names = result.names
        boxes = result.boxes
        classes = boxes.cls.tolist()
        scores = boxes.conf.tolist()
        xywhn = boxes.xywhn.tolist()

        detections = sorted(
            zip(classes, scores, xywhn), key=lambda x: x[1]
        )  # low to high confidence

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
            if not already_shown:
                img = cv2.imread(str(img_path))
                for cls_id, conf, box_xyxy in zip(
                    result.boxes.cls, result.boxes.conf, result.boxes.xyxy
                ):
                    cls_id = int(cls_id)
                    label = f"{names[cls_id]} {conf:.2f}"
                    x1, y1, x2, y2 = map(int, box_xyxy.tolist())

                    # draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # draw label above the box
                    text_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3
                    )[0]
                    text_x = x1
                    text_y = max(y1 - 10, text_size[1] + 10)

                    font_scale = 5
                    font_thickness = 5
                    text_color = (0, 255, 0)

                    cv2.putText(
                        img,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        text_color,
                        font_thickness,
                    )

                save_path = save_dir / img_path.name
                USE_ACDSEE = True
                if USE_ACDSEE:
                    subprocess.Popen(
                        f'"{acdsee_path}" "{str(result_img_path)}"',
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    img = cv2.imread(str(save_path))
                    if img is not None:
                        screen_width = 1200
                        if img.shape[1] > screen_width:
                            scale = screen_width / img.shape[1]
                            new_size = (
                                int(img.shape[1] * scale),
                                int(img.shape[0] * scale),
                            )
                            img = cv2.resize(img, new_size)

                        cv2.imshow(f"Prediction - {img_path.name}", img)
                        cv2.waitKey(1)

            print(
                f"❓ Review: {label} ({conf_pct}%) - (y = correct, w = wrong, s = all correct, c = skip, n = rest wrong, a = rest correct, b = rest wrong): ",
                end="",
            )

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
            elif choice == "w":
                label_file = wrong_label_dir / f"{img_path.stem}.txt"
                with open(label_file, "a") as f:
                    f.write(f"{int(cls_id)} {' '.join(map(str, box))}\n")
                print(f"❌ Marked as incorrect in: {label_file}")

                # === bulk labeling logic ===
        if skip_as_correct:
            label_file = active_label_dir / f"{img_path.stem}.txt"
            for cls_id, conf, box in detections:
                with open(label_file, "a") as f:
                    f.write(f"{int(cls_id)} {' '.join(map(str, box))}\n")
            print(f"✅ All detections saved as correct for: {img_path.name}")

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
