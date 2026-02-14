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
OUTPUT_DIR = "data/processed/greek_letters_classification"
IMG_SIZE = 128
SPLIT_RATIO = 0.8
# ---------------------

def parse_annotations_robust(file_path):
    """
    The STRICT parser we fixed earlier.
    Returns: boxes (list of 4-point tuples), labels (list of chars)
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
    Necessary for correct perspective transform.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def crop_and_warp(img, box):
    """
    Crops the OBB and warps it to a straight rectangle.
    """
    rect = order_points(box)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    # Resize to fixed size for training
    warped = cv2.resize(warped, (IMG_SIZE, IMG_SIZE))
    return warped

def prepare_dataset():
    # Setup Output Directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    # Get Files
    files = glob.glob(os.path.join(INPUT_ANN_DIR, "*_Rotation1_300dpi_letters.txt"))
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
                    # Crop and Deskew
                    crop = crop_and_warp(img, box)
                    
                    # Create label folder if not exists
                    label_dir = os.path.join(OUTPUT_DIR, split, label)
                    os.makedirs(label_dir, exist_ok=True)
                    
                    # Save Crop
                    save_name = f"{base_name}_{i}.jpg"
                    cv2.imwrite(os.path.join(label_dir, save_name), crop)
                    
                except Exception as e:
                    # Occasional perspective math errors on tiny boxes can be skipped
                    continue

    print("Done! You can now check the output folder.")

if __name__ == "__main__":
    prepare_dataset()