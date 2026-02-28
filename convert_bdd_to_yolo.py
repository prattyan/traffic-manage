import os
import json
from PIL import Image

# ============================
# SELECT SPLIT HERE
# ============================

SPLIT = "train"  # change to "train" or "test" when needed

# ============================
# PATHS
# ============================

BASE_PATH = r"C:\Users\Admin\Desktop\Apertre\traffic-manage"

LABELS_PATH = os.path.join(BASE_PATH, "100k", SPLIT)
IMAGE_PATH = os.path.join(BASE_PATH, "bdd100k", SPLIT)
OUTPUT_PATH = os.path.join(BASE_PATH, "yolo_labels", SPLIT)

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================
# VEHICLE CLASSES
# ============================

CLASSES = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "motorcycle": 3,
    "bike": 4
}

converted_files = 0

# ============================
# CONVERSION LOOP
# ============================

for file in os.listdir(LABELS_PATH):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(LABELS_PATH, file)

    with open(json_path, "r") as f:
        data = json.load(f)

    base_name = data.get("name")

    if not base_name:
        continue

    for frame in data.get("frames", []):

        timestamp = frame.get("timestamp")

        if timestamp is None:
            continue

        # Convert timestamp (10000 → 0000100)
        frame_number = int(timestamp / 100)
        frame_str = f"{frame_number:07d}"

        image_name = f"{base_name}-{frame_str}.jpg"
        image_path = os.path.join(IMAGE_PATH, image_name)

        if not os.path.exists(image_path):
            print("Image not found:", image_name)
            continue

        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except:
            continue

        yolo_lines = []

        for obj in frame.get("objects", []):

            category = obj.get("category")

            if category not in CLASSES:
                continue

            box = obj.get("box2d")

            if not box:
                continue

            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]

            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            class_id = CLASSES[category]

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        if yolo_lines:
            output_file = os.path.join(
                OUTPUT_PATH,
                image_name.replace(".jpg", ".txt")
            )

            with open(output_file, "w") as out:
                out.write("\n".join(yolo_lines))

            converted_files += 1

print(f"✅ Conversion Completed! {converted_files} files converted.")