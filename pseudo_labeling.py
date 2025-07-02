import cv2
from ultralytics import YOLO
import os
from glob import glob

# Load model (Pose Estimation model)
model = YOLO("/home/work/.doheon114/ICU_Anomaly/ultralytics/yolo11x-pose.pt")

# Collect all image paths
image_root_dir = "/home/work/.doheon114/ICU_Anomaly/extracted_frames"
image_paths = glob(os.path.join(image_root_dir, "**", "*.jpg"), recursive=True)  # search for jpg files
# Output root directory
pseudo_label_root = "/home/work/.doheon114/ICU_Anomaly/pseudo_labels"

for image_path in image_paths:
    print(f"Processing: {image_path}")
    
    # Set output directory based on relative path
    rel_path = os.path.relpath(image_path, image_root_dir)
    image_dir = os.path.dirname(rel_path)  # subdirectory path containing the image
    image_name = os.path.splitext(os.path.basename(rel_path))[0]  # image filename without extension
    
    output_label_dir = os.path.join(pseudo_label_root, image_dir)
    os.makedirs(output_label_dir, exist_ok=True)

    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to read image: {image_path}")
        continue

    results = model.predict(frame, iou=0.35, conf=0.4)


    for result in results:
        height, width, _ = frame.shape
        label_lines = []

        boxes = result.boxes.xyxy if result.boxes is not None else []
        classes = result.boxes.cls if result.boxes is not None else []
        keypoints = result.keypoints.xy if result.keypoints is not None else []

        for i in range(len(keypoints)):
            if len(boxes) <= i:
                continue
            x1, y1, x2, y2 = boxes[i]
            cls = int(classes[i]) if len(classes) > i else 0
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            kp_set = keypoints[i]
            keypoint_str = ''
            for x, y in kp_set:
                x_norm = x.item() / width
                y_norm = y.item() / height
                keypoint_str += f" {x_norm:.6f} {y_norm:.6f}"

            label_line = f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}" + keypoint_str
            label_lines.append(label_line)

        label_path = os.path.join(output_label_dir, f"{image_name}.txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
        result.save(filename=os.path.join(output_label_dir, f"{image_name}.jpg"))

    print(f"Saved pseudo label to: {os.path.join(output_label_dir, image_name)}.txt")
