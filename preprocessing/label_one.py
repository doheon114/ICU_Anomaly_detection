import os
import shutil

labels_src_root = "/home/work/.doheon114/ICU_Anomaly/pseudo_labels/labels"
labels_dst_root = "/home/work/.doheon114/ICU_Anomaly/pseudo_labels/labels_one"
images_src_root = "/home/work/.doheon114/ICU_Anomaly/pseudo_labels/images"
images_dst_root = "/home/work/.doheon114/ICU_Anomaly/pseudo_labels/images_one"

for dirpath, _, filenames in os.walk(labels_src_root):
    for filename in filenames:
        if filename.endswith(".txt"):
            label_src_path = os.path.join(dirpath, filename)
            with open(label_src_path, "r") as f:
                lines = f.readlines()
            if len(lines) == 1:
                # labels_one에 복사
                rel_path = os.path.relpath(label_src_path, labels_src_root)
                label_dst_path = os.path.join(labels_dst_root, rel_path)
                os.makedirs(os.path.dirname(label_dst_path), exist_ok=True)
                shutil.copy2(label_src_path, label_dst_path)

                # images_one에 이미지 복사 (확장자만 .txt → .jpg로 변경)
                image_rel_path = rel_path[:-4] + ".png"  # .txt → .png
                image_src_path = os.path.join(images_src_root, image_rel_path)
                image_dst_path = os.path.join(images_dst_root, image_rel_path)
                if os.path.exists(image_src_path):
                    os.makedirs(os.path.dirname(image_dst_path), exist_ok=True)
                    shutil.copy2(image_src_path, image_dst_path)
                else:
                    print(f"이미지 파일 없음: {image_src_path}")
