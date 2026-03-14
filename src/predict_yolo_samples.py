import os
import random
import glob
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'runs/obb/train3/weights/best.pt'
SOURCE_DIR = 'data/processed/greek_letters_obb/images/val'  # Update this to your actual test image folder
NUM_SAMPLES = 10
# ---------------------

# 1. Load your best model
model = YOLO(MODEL_PATH)

# 2. Get list of all images (jpg, png, etc.)
# Recursive search for common image formats
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
all_images = []
for ext in image_extensions:
    all_images.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))

# 3. Select 10 random images
if len(all_images) > NUM_SAMPLES:
    selected_images = random.sample(all_images, NUM_SAMPLES)
else:
    # If fewer than 10 images exist, take them all
    selected_images = all_images
    print(f"Warning: Only found {len(all_images)} images. Using all of them.")

print(f"Running inference on {len(selected_images)} images...")

# 4. Run inference
results = model.predict(
    source=selected_images,   # Pass the LIST of paths, not the folder string
    save=True,                # Save the images with drawn boxes
    imgsz=640,                # Must match training size
    conf=0.25,                # Only show boxes with > 25% confidence
    line_width=2              # Makes the boxes slightly thicker for easier viewing
)

print("Inference complete. Check 'runs/obb/predict/' for results.")