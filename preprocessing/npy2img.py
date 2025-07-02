import numpy as np
import cv2 as cv

# npy 파일 경로
npy_path = 'ICU_Anomaly/pseudo_labels_one/images_depth/train/SICU2/06090612/06-09_17-56-13/second_00073.npy'
# 저장할 png 파일 경로
png_path = 'second_00073.png'

# 거리 범위 
min_distance = 0.3
max_distance = 3.0

# npy 파일 로드
depth_data = np.load(npy_path)

# depth map 변환 
depth_map = ((depth_data - min_distance) * (255 / (max_distance - min_distance)))
depth_map = np.clip(depth_map, 0, 255).astype(np.uint8)

# 3채널로 변환 
depth_map_color = np.stack([depth_map, depth_map, depth_map], axis=-1)

# png로 저장
cv.imwrite(png_path, depth_map_color)
