from pathlib import Path
import shutil

root_dir = Path('/home/work/.doheon114/ICU_Anomaly/pseudo_labels/labels')

# 1단계: 파일 이동
for file_path in root_dir.rglob('*'):
    if file_path.is_file() and '_rgb' in file_path.parent.name:
        new_parent = file_path.parent.parent / file_path.parent.name.replace('_rgb', '')
        new_parent.mkdir(parents=True, exist_ok=True)
        new_file_path = new_parent / file_path.name
        try:
            shutil.move(str(file_path), str(new_file_path))
            print(f"Moved: {file_path} → {new_file_path}")
        except Exception as e:
            print(f"Error moving {file_path}: {e}")

# 2단계: 빈 `_rgb` 폴더 삭제
for dir_path in root_dir.rglob('*_rgb'):
    if dir_path.is_dir():
        try:
            dir_path.rmdir()  # 비어있을 때만 삭제 가능
            print(f"Deleted empty folder: {dir_path}")
        except OSError as e:
            print(f"Cannot delete non-empty folder: {dir_path} — {e}")
