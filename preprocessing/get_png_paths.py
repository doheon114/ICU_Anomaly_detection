import os
import glob

# SICU4 디렉토리의 절대 경로
base_dir = '/home/work/.doheon114/ICU_Anomaly/pseudo_labels/images/train'

# glob을 사용하여 모든 png 파일의 경로를 찾습니다
png_files = glob.glob(os.path.join(base_dir, '**/*.png'), recursive=True)

# 결과를 train.txt 파일에 저장합니다
output_path = os.path.join('/home/work/.doheon114/ICU_Anomaly/pseudo_labels_one', 'train.txt')
with open(output_path, 'w') as f:
    for png_path in png_files:
        f.write(os.path.abspath(png_path) + '\n')

print(f'총 {len(png_files)}개의 PNG 파일 경로가 {output_path}에 저장되었습니다.')
