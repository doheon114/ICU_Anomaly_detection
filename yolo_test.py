from ultralytics import YOLO

# Load a model
model = YOLO("/home/work/.doheon114/ICU_Anomaly/ultralytics/yolo11m-pose.pt")  # load an official model
model = YOLO("/home/work/.doheon114/ICU_Anomaly/results/default_baseline/train/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)  # a list contains map50-95 of each category
print(metrics.pose.map)  # map50-95(P)
print(metrics.pose.map50)  # map50(P)
print(metrics.pose.map75)  # map75(P)
print(metrics.pose.maps)  # a list contains map50-95(P) of each category
