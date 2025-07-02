from ultralytics import YOLO

# Load a model
model = YOLO("/home/work/.doheon114/ICU_Anomaly/ultralytics/yolo11m-pose.pt")  

# Train the model
results = model.train(
    # Data configuration
    data="/home/work/.doheon114/ICU_Anomaly/pseudo_labels/ICU.yaml",
    
    
    # Training resources
    workers=40,
    batch=128,
    device='1',
    
    # Training parameters
    epochs=70,
    lr0=0.01,
    lrf=0.01,
    optimizer='auto',
    patience=10,
    
    # Model configuration
    imgsz=640,
    dropout=0.1,
    
    # Loss weights
    box=7.5,
    kobj=2.0,
    pose=12.0,
    cls=0.5,
    dfl=1.5,
    
    # Augmentation
    augment=False,
    mosaic=1.0,
    
    # Saving and visualization
    project="default_baseline_one_real",
    plots=True,
    save_period=5,

)
