from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("//home/work/.doheon114/ICU_Anomaly/results/default_baseline/train/weights/best.pt")

# Define path to video file
source = "/home/work/.doheon114/ICU_Anomaly/dataset/SICU4/depth/10301031/10-30_16-02-52_depth.mp4"

# Run inference on the source
results = model(source, save=True)  # generator of Results objects
