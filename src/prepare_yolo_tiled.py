import os
import glob
import cv2
import numpy as np
from shapely.geometry import Polygon, box
from tqdm import tqdm
import shutil

# --- CONFIGURATION ---
INPUT_IMG_DIR = "data/raw/Images"
INPUT_ANN_DIR = "data/raw/Annotations"
OUTPUT_DIR = "data/processed/greek_letters_obb"  # Where YOLO data will go

TILE_SIZE = 640
STRIDE = 512  # 128px overlap (640 - 512 = 128)
IOA_THRESHOLD = 0.7  # Keep box if 70% of it is visible in the tile

# ---------------------

def parse_obb_annotations(file_path):
    """
    Parses your dataset format into a list of Polygons.
    Returns: List of [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    polygons = []
    if not os.path.exists(file_path):
        return polygons
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find where boxes start
    start_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("# Boxes"):
            start_idx = i + 1
            break
            
    if start_idx != -1:
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if line.startswith("#"): break # Stop at next section
            
            parts = line.split()
            if len(parts) == 8:
                try:
                    coords = [float(p) for p in parts]
                    # Create list of tuples for Shapely
                    poly_coords = [
                        (coords[0], coords[1]), 
                        (coords[2], coords[3]), 
                        (coords[4], coords[5]), 
                        (coords[6], coords[7])
                    ]
                    polygons.append(poly_coords)
                except ValueError:
                    continue
    return polygons

def save_yolo_label(file_path, polygons, img_width, img_height):
    """
    Saves annotations in YOLO OBB format:
    class_index x1 y1 x2 y2 x3 y3 x4 y4 (Normalized 0-1)
    """
    with open(file_path, 'w') as f:
        for poly in polygons:
            # Flatten coordinates
            flat_coords = []
            for (x, y) in poly:
                # Normalize coordinates relative to tile size
                nx = max(0, min(x / img_width, 1.0))
                ny = max(0, min(y / img_height, 1.0))
                flat_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
            
            # Class 0 is 'Letter'
            line = f"0 {' '.join(flat_coords)}\n"
            f.write(line)

def process_dataset():
    # Create directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    # Get file list (Rotation1 files as anchors)
    files = glob.glob(os.path.join(INPUT_ANN_DIR, "*_Rotation1_300dpi_letters.txt"))
    
    # Simple split: 90% train, 10% val
    np.random.shuffle(files)
    split_idx = int(len(files) * 0.9)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    datasets = {
        'train': train_files,
        'val': val_files
    }

    print(f"Starting Tiling process... ({TILE_SIZE}x{TILE_SIZE} tiles, stride {STRIDE})")

    for split, file_list in datasets.items():
        print(f"Processing {split} set...")
        
        for ann_path in tqdm(file_list):
            base_name = os.path.basename(ann_path).replace("_letters.txt", "")
            img_name = base_name + ".png"
            img_path = os.path.join(INPUT_IMG_DIR, img_name)
            
            if not os.path.exists(img_path):
                continue

            # Load Image
            img = cv2.imread(img_path)
            if img is None: continue
            h_img, w_img, _ = img.shape

            # Parse Boxes
            boxes_raw = parse_obb_annotations(ann_path)
            # Convert to Shapely polygons for math
            shapely_boxes = [Polygon(b) for b in boxes_raw if len(b) >= 3]

            # Sliding Window
            tile_count = 0
            for y in range(0, h_img, STRIDE):
                for x in range(0, w_img, STRIDE):
                    # Define Tile Geometry
                    x_end = min(x + TILE_SIZE, w_img)
                    y_end = min(y + TILE_SIZE, h_img)
                    
                    # Correct actual width/height for edge tiles
                    cur_w = x_end - x
                    cur_h = y_end - y
                    
                    # Create Tile Polygon
                    tile_poly = box(x, y, x_end, y_end)
                    
                    # Find Intersecting Boxes
                    valid_tile_boxes = []
                    
                    for idx, s_box in enumerate(shapely_boxes):
                        if not s_box.is_valid: continue
                        
                        # Intersection Logic
                        try:
                            intersection = s_box.intersection(tile_poly)
                            if intersection.is_empty:
                                continue
                                
                            # Calculate IoA (Intersection over Area of the BOX)
                            # How much of the box is visible?
                            ioa = intersection.area / s_box.area
                            
                            if ioa >= IOA_THRESHOLD:
                                # We keep this box. 
                                # Now we need coordinates relative to the tile (0,0)
                                # We use the intersection polygon (clipped to edge)
                                # OR the original box shifted (if you want partial boxes extending out)
                                
                                # Use intersection exterior coords (clipped)
                                if intersection.geom_type == 'Polygon':
                                    coords = list(intersection.exterior.coords)[:-1] # Drop last duplicate point
                                    
                                    # Ensure we have 4 points for OBB (YOLO technically supports polygons, 
                                    # but 4 points is standard. If clipped, it might be 3 or 5 points.
                                    # Simplification: Use minimum rotated rectangle of the intersection
                                    rect = intersection.minimum_rotated_rectangle
                                    coords = list(rect.exterior.coords)[:-1]
                                    
                                    # Shift coordinates to tile local space
                                    local_coords = []
                                    for (px, py) in coords:
                                        local_coords.append((px - x, py - y))
                                        
                                    valid_tile_boxes.append(local_coords)
                        except Exception as e:
                            pass

                    # Only save tile if it contains annotations (optional, saves disk space)
                    if len(valid_tile_boxes) > 0:
                        tile_name = f"{base_name}_{x}_{y}"
                        
                        # Save Image Tile
                        tile_img = img[y:y_end, x:x_end]
                        # Pad edge tiles to be full TILE_SIZE (Optional but recommended for batching)
                        if cur_h < TILE_SIZE or cur_w < TILE_SIZE:
                            padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                            padded[:cur_h, :cur_w] = tile_img
                            tile_img = padded
                            
                        save_img_path = os.path.join(OUTPUT_DIR, 'images', split, tile_name + ".jpg")
                        cv2.imwrite(save_img_path, tile_img)
                        
                        # Save Label
                        save_lbl_path = os.path.join(OUTPUT_DIR, 'labels', split, tile_name + ".txt")
                        save_yolo_label(save_lbl_path, valid_tile_boxes, TILE_SIZE, TILE_SIZE)
                        
                        tile_count += 1
    
    print("\nDataset preparation complete.")
    print(f"Create a YAML file pointing to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    process_dataset()