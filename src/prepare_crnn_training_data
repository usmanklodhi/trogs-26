import os
import cv2
import numpy as np
import glob
import random
import shutil
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_IMG_DIR = "data/raw/Images"
INPUT_ANN_DIR = "data/raw/Annotations"
OUTPUT_DIR = "data/crnn_training_data"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def order_points(pts):
    """Orders points: top-left, top-right, bottom-right, bottom-left for perspective warping."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def parse_lines_from_annotations(file_path):
    """Parses annotations, skips asterisks for box counting, but keeps them in labels."""
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    boxes = []
    transcript_start_idx = -1

    # Extract 8-Point Boxes
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith("# Boxes"):
            j = i + 1
            while j < len(lines):
                if lines[j].strip().startswith("# Transcript"):
                    transcript_start_idx = j + 1
                    break
                parts = lines[j].strip().split()
                if len(parts) == 8:
                    try:
                        c = [float(p) for p in parts]
                        poly = np.array([
                            [c[0], c[1]], [c[2], c[3]], 
                            [c[4], c[5]], [c[6], c[7]]
                        ], dtype=np.float32)
                        boxes.append(poly)
                    except ValueError: pass
                j += 1
            break

    line_data = []
    box_index = 0

    # Extract Transcript and Group
    if transcript_start_idx != -1:
        for i in range(transcript_start_idx, len(lines)):
            line_content = lines[i].strip()
            
            if line_content.startswith("# Lines"): 
                break
                
            clean_text = "".join([c for c in line_content if c not in (' ', '\t')])
            chars_with_boxes = [c for c in clean_text if c != '*']
            num_boxes_to_consume = len(chars_with_boxes)
            
            if num_boxes_to_consume == 0:
                continue
                
            line_boxes = boxes[box_index : box_index + num_boxes_to_consume]
            box_index += num_boxes_to_consume
            
            if not line_boxes:
                continue

            all_points = np.vstack(line_boxes) 
            
            line_data.append({
                'text': clean_text,
                'points': all_points 
            })

    return line_data

# ==========================================
# 3. MAIN EXTRACTION PIPELINE
# ==========================================
def process_crnn_dataset():
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning up old {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
        
    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    labels_path = os.path.join(OUTPUT_DIR, "labels.txt")
    labels_file = open(labels_path, "w", encoding="utf-8")
    
    ann_files = glob.glob(os.path.join(INPUT_ANN_DIR, "*_letters.txt"))
    print(f"Found {len(ann_files)} annotation files. Starting extraction...")
    
    total_lines = 0
    error_count = 0
    
    for ann_path in tqdm(ann_files):
        base_name = os.path.basename(ann_path).replace("_letters.txt", "")
        img_path = os.path.join(INPUT_IMG_DIR, base_name + ".png")
        
        if not os.path.exists(img_path):
            error_count += 1
            continue
            
        img = cv2.imread(img_path)
        if img is None: 
            error_count += 1
            continue
        
        lines_data = parse_lines_from_annotations(ann_path)
        
        for idx, line in enumerate(lines_data):
            pts = line['points']
            
            # 1. OBB Calculation
            rect = cv2.minAreaRect(pts)
            (cx, cy), (w, h), angle = rect

            if h > w:
                w, h = h, w
                angle -= 90.0
                
            h = max(1.0, h) # Prevent zero division

            # 2. Apply +T Data Augmentation
            extension = random.choice([0, int(h // 2), int(h)])
            padded_w = w + (2 * extension)

            padded_rect = ((cx, cy), (padded_w, h), angle)
            box = cv2.boxPoints(padded_rect)
            src_pts = order_points(box)

            # 3. Perspective Warp
            target_h = 32
            scale = target_h / float(h)
            target_w = max(1, int(padded_w * scale))

            dst_pts = np.array([
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_crop = cv2.warpPerspective(img, M, (target_w, target_h), borderValue=(0, 0, 0))
            
            # Save Image and Label
            output_filename = f"{base_name}_line{idx:03d}.jpg"
            output_filepath = os.path.join(images_dir, output_filename)
            cv2.imwrite(output_filepath, warped_crop)
            
            labels_file.write(f"{output_filename}\t{line['text']}\n")
            total_lines += 1
            
    labels_file.close()
    
    print("\n--- Extraction Summary ---")
    print(f"Total line crops generated: {total_lines}")
    print(f"Files skipped (missing/unreadable): {error_count}")
    print(f"Dataset ready at: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_crnn_dataset()