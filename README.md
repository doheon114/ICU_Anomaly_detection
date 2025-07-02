# ICU_Anomaly

This repository provides preprocessing, pseudo-labeling, and pose-based anomaly detection training/inference scripts for ICU patient monitoring using YOLO pose models.

## Folder Structure

```
ICU_Anomaly/
├── preprocessing/         # Preprocessing scripts (frame extraction, augmentation, etc.)
├── pseudo_labeling.py     # Pseudo-label generation using YOLO pose model
├── yolo_train.py          # Training script for pose-based anomaly detection
├── yolo_inference.py      # Inference script for video
├── yolo_test.py           # Evaluation/validation script
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignore large data, models, cache, etc.
```

## Preprocessing
- All scripts in `preprocessing/` are for data preparation (frame extraction, augmentation, etc).

## Pseudo Labeling
- `pseudo_labeling.py` generates pseudo-labels using a YOLO pose model for all images in a directory.

## Training
- `yolo_train.py` trains a YOLO pose model using the generated pseudo-labels.

## Inference
- `yolo_inference.py` runs inference on a video using a trained model.

## Evaluation
- `yolo_test.py` evaluates the trained model and prints mAP/pose metrics.

## Installation
```bash
pip install -r requirements.txt
```

## Notes
- Model weights and large datasets are **not** included in this repository.
- Please download or prepare your own data and models as needed.
- For more details, see each script's comments. 
