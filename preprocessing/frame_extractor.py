from pathlib import Path
import cv2
import re

# 경로 설정
label_root = Path('/home/work/.doheon114/ICU_Anomaly/pseudo_labels/labels')
dataset_root = Path('/home/work/.doheon114/ICU_Anomaly/dataset')
image_root = Path('/home/work/.doheon114/ICU_Anomaly/pseudo_labels/images')

# 모든 txt 파일 찾기
txt_files = list(label_root.rglob('*.txt'))

for txt_path in txt_files:
    # txt 파일의 상대 경로에서 필요한 정보 추출
    relative_path = txt_path.relative_to(label_root)
    parts = list(relative_path.parts)
    
    # second_XXXXX.txt에서 초 추출
    second_match = re.match(r'second_(\d+)\.txt', parts[-1])
    if not second_match:
        continue
        
    second = int(second_match.group(1))
    
    # 비디오 파일 경로 구성
    # train/SICU2/06090612/06-09_17-56-13/second_00010.txt
    # -> SICU2/depth/06090612/06-09_17-56-13_depth.mp4
    sicu_folder = parts[1]  # SICU2
    date_folder = parts[2]  # 06090612
    time_folder = parts[3]  # 06-09_17-56-13
    
    video_path = dataset_root / sicu_folder / 'depth' / date_folder / f'{time_folder}_depth.mp4'
    
    if not video_path.exists():
        print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        continue
        
    # 이미지 저장 경로 구성
    image_save_dir = image_root / '/'.join(parts[:-1])
    image_save_dir.mkdir(parents=True, exist_ok=True)
    image_save_path = image_save_dir / f'{parts[-1].replace(".txt", ".png")} '
    
    # 비디오에서 프레임 추출
    cap = cv2.VideoCapture(str(video_path))
    fps = 15 if second == 1 else cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * (second))
    
    # 원하는 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if ret:
        # 이미지 저장
        cv2.imwrite(str(image_save_path), frame)
        print(f"이미지 저장 완료: {image_save_path}")
    else:
        print(f"프레임 추출 실패: {video_path} - {second}초")
    
    cap.release()

print("모든 작업이 완료되었습니다.") 