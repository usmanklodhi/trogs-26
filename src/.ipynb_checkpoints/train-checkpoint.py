from ultralytics import YOLO
# Untested so far


# Load the model
model = YOLO('yolo11n-obb.pt')

# Train with specific augmentations for Greek Inscriptions
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    
    # --- Geometric Augs ---
    degrees=15.0,      # Rotate +/- 15 degrees
    fliplr=0.0,        # DISABLE horizontal flip (Letters aren't symmetric)
    flipud=0.0,        # DISABLE vertical flip
    scale=0.5,         # Zoom +/- 50%
    shear=2.0,         # Slight shear to simulate perspective tilt
    
    # --- Photometric Augs ---
    hsv_h=0.015,       # Minimal color shift
    hsv_s=0.2,         # Low saturation shift
    hsv_v=0.5,         # HIGH brightness shift (Light/Dark stone)
    
    # # --- Advanced ---
    # mosaic=1.0,        # Try 1.0 first. If accuracy is bad, set to 0.0
    # close_mosaic=10,   # Disable mosaic for the last 10 epochs (Standard practice)
)