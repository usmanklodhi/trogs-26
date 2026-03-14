from ultralytics import YOLO

# 1. Load the trained weights
# Adjust the path 'runs/obb/train...' if your folder has a number like 'train2'
model = YOLO('runs/obb/train3/weights/best.pt')

# 2. Run Validation
# This uses the 'val' images defined in your data.yaml
metrics = model.val(
    data='/home/hpc/iwi5/iwi5340h/trogs-26/data/processed/data.yaml',
    split='val',   # explicit split
    imgsz=640,     # Must match your training size
    batch=16,
    device='cpu'       # Use GPU
)

# 3. Print Key Metrics
print(f"mAP50 (Oriented): {metrics.box.map50}")       # Accuracy at 50% IoU
print(f"mAP50-95 (Oriented): {metrics.box.map}")      # Accuracy averaged over 50-95% IoU