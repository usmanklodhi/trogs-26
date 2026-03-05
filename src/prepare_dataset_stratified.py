import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import shutil
import random

# --- CONFIGURATION ---
INPUT_IMG_DIR = "data/raw/Images"       # Point this to where the .png files are
INPUT_ANN_DIR = "data/raw/Annotations"  # Point this to where the .txt files are
OUTPUT_DIR = "data/processed/greek_letters_classification_stratified"
IMG_SIZE = 128
SPLIT_RATIO = 0.8
# ---------------------

def parse_annotations_robust(file_path):
    """
    Parses the .txt file to extract boxes and transcript labels.
    """
    boxes = []
    
    if not os.path.exists(file_path):
        return boxes, []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    transcript_start_idx = -1
    
    # 1. Parse Boxes
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith("# Boxes"):
            j = i + 1
            while j < len(lines):
                if lines[j].strip().startswith("# Transcript"):
                    break
                parts = lines[j].strip().split()
                if len(parts) == 8:
                    try:
                        c = [float(p) for p in parts]
                        # Store as list of points [[x,y], [x,y]...]
                        poly = np.array([
                            [c[0], c[1]], [c[2], c[3]], 
                            [c[4], c[5]], [c[6], c[7]]
                        ], dtype=np.float32)
                        boxes.append(poly)
                    except: pass
                j += 1
        elif line_stripped.startswith("# Transcript"):
            transcript_start_idx = i + 1

    # 2. Parse Transcript
    clean_labels = []
    if transcript_start_idx != -1:
        for i in range(transcript_start_idx, len(lines)):
            line_content = lines[i]
            if line_content.strip().startswith("# Lines"): break
            
            for char in line_content:
                if char in ('*', ' ', '\n', '\r', '\t'): continue
                clean_labels.append(char)

    # 3. Validation Truncation
    min_len = min(len(boxes), len(clean_labels))
    return boxes[:min_len], clean_labels[:min_len]

def order_points(pts):
    """
    Orders coordinates: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def crop_and_warp_square_padding(img, box):
    """
    ASPECT RATIO PRESERVING CROP:
    1. Finds the center of the rotated box.
    2. Finds the maximum dimension (width vs height).
    3. Expands the box on the original image to be a square of that max size.
    4. Warps that square region to 128x128.
    """
    rect = order_points(box)
    (tl, tr, br, bl) = rect

    # 1. Compute current Width and Height of the rotated box
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    current_w = max(widthA, widthB)

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    current_h = max(heightA, heightB)
    
    # 2. Determine the Square Size (Largest side)
    max_side = max(int(current_w), int(current_h))
    
    # 3. Calculate Center of the box
    center = np.mean(rect, axis=0)
    
    # 4. Calculate Unit Vectors for Orientation
    # Top edge direction (Width Vector)
    vec_w = (tr - tl)
    norm_w = np.linalg.norm(vec_w)
    unit_w = vec_w / (norm_w + 1e-6) # Avoid division by zero
    
    # Left edge direction (Height Vector)
    vec_h = (bl - tl)
    norm_h = np.linalg.norm(vec_h)
    unit_h = vec_h / (norm_h + 1e-6)
    
    # 5. Calculate New Corners for the Square Box
    # We expand from the center using the unit vectors and half the max_side
    half = max_side / 2.0
    
    new_tl = center - (unit_w * half) - (unit_h * half)
    new_tr = center + (unit_w * half) - (unit_h * half)
    new_br = center + (unit_w * half) + (unit_h * half)
    new_bl = center - (unit_w * half) + (unit_h * half)
    
    src_square = np.array([new_tl, new_tr, new_br, new_bl], dtype="float32")
    
    # 6. Destination is a simple upright 128x128 square
    dst = np.array([
        [0, 0],
        [IMG_SIZE - 1, 0],
        [IMG_SIZE - 1, IMG_SIZE - 1],
        [0, IMG_SIZE - 1]], dtype="float32")

    # 7. Warp
    M = cv2.getPerspectiveTransform(src_square, dst)
    # borderValue=(0,0,0) fills with black if the box goes off the image edge
    warped = cv2.warpPerspective(img, M, (IMG_SIZE, IMG_SIZE), borderValue=(0, 0, 0))
    
    return warped

def prepare_dataset():
    # Setup Output Directories
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning up old {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    # Get Files
    files = glob.glob(os.path.join(INPUT_ANN_DIR, "*_Rotation1_300dpi_letters.txt"))
    # Shuffle for random split
    random.shuffle(files)
    
    split_idx = int(len(files) * SPLIT_RATIO)
    splits = {'train': files[:split_idx], 'val': files[split_idx:]}

    print(f"Processing {len(files)} files into {OUTPUT_DIR}...")

    for split, file_list in splits.items():
        print(f"Processing {split}...")
        
        for ann_path in tqdm(file_list):
            base_name = os.path.basename(ann_path).replace("_letters.txt", "")
            img_path = os.path.join(INPUT_IMG_DIR, base_name + ".png")
            
            if not os.path.exists(img_path):
                continue

            # Load Image
            img = cv2.imread(img_path)
            if img is None: continue

            # Get Data
            boxes, labels = parse_annotations_robust(ann_path)

            for i, (box, label) in enumerate(zip(boxes, labels)):
                try:
                    # New cropping logic (Aspect Ratio Preserved)
                    crop = crop_and_warp_square_padding(img, box)
                    
                    # Create label folder
                    label_dir = os.path.join(OUTPUT_DIR, split, label)
                    os.makedirs(label_dir, exist_ok=True)
                    
                    # Save Crop
                    save_name = f"{base_name}_{i}.jpg"
                    cv2.imwrite(os.path.join(label_dir, save_name), crop)
                    
                except Exception as e:
                    continue

    print("Done! Dataset prepared with aspect-ratio preserving crops.")

if __name__ == "__main__":
    prepare_dataset()