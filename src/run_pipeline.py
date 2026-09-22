import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from PIL import Image
from torchvision import transforms, models
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ==========================================
# 1. SETUP AND CONFIGURATION
# ==========================================
YOLO_MODEL_PATH = '../runs/obb/train3/weights/best.pt'
CLS_MODEL_PATH = '../models/greek_classifier_resnet18_20260214_212940.pth'
TRAIN_DIR = '../data/processed/greek_letters_classification/train'
PROXIMITY_THRESHOLD = 50 

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

def load_pipeline():
    print(f"Initializing models on {DEVICE}...")
    CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=YOLO_MODEL_PATH,
        confidence_threshold=0.25,
        device=DEVICE,
    )

    classifier = models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, len(CLASS_NAMES))
    classifier.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
    classifier.to(DEVICE)
    classifier.eval()

    cls_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Models ready.")
    return detection_model, classifier, cls_transforms, CLASS_NAMES

# ==========================================
# 2. TOURNAMENT INFERENCE
# ==========================================
def run_tournament(img1_path, img2_path, detector, classifier, transforms, class_names):
    print("Running detection on Image 1...")
    res1 = get_sliced_prediction(img1_path, detector, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, postprocess_type="NMS")
    boxes1 = res1.object_prediction_list

    img1_cv2 = cv2.imread(img1_path)
    h1, w1, _ = img1_cv2.shape

    if img2_path is None:
        print("No Image 2 provided, processing single image...")
        boxes2 = []
        img2_cv2 = None
        h2 = w2 = 0
    else:
        print("Running detection on Image 2...")
        res2 = get_sliced_prediction(img2_path, detector, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, postprocess_type="NMS")
        boxes2 = res2.object_prediction_list

        img2_cv2 = cv2.imread(img2_path)
        h2, w2, _ = img2_cv2.shape

    final_results = []
    
    print("Running Proximity Matching & Logit Averaging Fusion...")
    for b1 in boxes1:
        x1_min, y1_min, x1_max, y1_max = map(int, b1.bbox.to_xyxy())
        cx1, cy1 = (x1_min + x1_max) / 2, (y1_min + y1_max) / 2
        
        closest_b2 = None
        min_dist = float('inf')
        
        for b2 in boxes2:
            x2_min, y2_min, x2_max, y2_max = map(int, b2.bbox.to_xyxy())
            cx2, cy2 = (x2_min + x2_max) / 2, (y2_min + y2_max) / 2
            dist = math.hypot(cx1 - cx2, cy1 - cy2)
            
            if dist < min_dist:
                min_dist = dist
                closest_b2 = b2
                best_coords2 = (x2_min, y2_min, x2_max, y2_max)
                
        if closest_b2 and min_dist < PROXIMITY_THRESHOLD:
            x2_min, y2_min, x2_max, y2_max = best_coords2
            
            # Safe Crop 1
            x1_min_c, y1_min_c = max(0, x1_min), max(0, y1_min)
            x1_max_c, y1_max_c = min(w1, x1_max), min(h1, y1_max)
            crop1 = img1_cv2[y1_min_c:y1_max_c, x1_min_c:x1_max_c]
            
            # Safe Crop 2
            x2_min_c, y2_min_c = max(0, x2_min), max(0, y2_min)
            x2_max_c, y2_max_c = min(w2, x2_max), min(h2, y2_max)
            crop2 = img2_cv2[y2_min_c:y2_max_c, x2_min_c:x2_max_c]
            
            if crop1.size == 0 or crop2.size == 0: continue
            
            t1 = transforms(Image.fromarray(cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
            t2 = transforms(Image.fromarray(cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out1, out2 = classifier(t1), classifier(t2)
                
                # --- LOGIT AVERAGING (LATE FUSION) ---
                avg_logits = (out1 + out2) / 2.0
                avg_probs = F.softmax(avg_logits, dim=1)
                best_conf_tensor, best_p_tensor = torch.max(avg_probs, 1)
                
                best_label = class_names[best_p_tensor.item()]
                best_conf = best_conf_tensor.item()
                
            final_results.append({
                'x1': x1_min, 'y1': y1_min, 'x2': x1_max, 'y2': y1_max, 
                'label': best_label, 'conf': best_conf
            })

        else:
            x1_min_c, y1_min_c = max(0, x1_min), max(0, y1_min)
            x1_max_c, y1_max_c = min(w1, x1_max), min(h1, y1_max)
            crop1 = img1_cv2[y1_min_c:y1_max_c, x1_min_c:x1_max_c]
            if crop1.size == 0: continue
            
            t1 = transforms(Image.fromarray(cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                conf1, p1 = torch.max(F.softmax(classifier(t1), dim=1), 1)
                
            final_results.append({
                'x1': x1_min, 'y1': y1_min, 'x2': x1_max, 'y2': y1_max, 
                'label': class_names[p1.item()], 'conf': conf1.item()
            })
            
    return final_results

# ==========================================
# 3. GEOMETRIC RECONSTRUCTION (From Paper)
# ==========================================
def calculate_row_bbox(row_boxes):
    return (min(b['x1'] for b in row_boxes), min(b['y1'] for b in row_boxes),
            max(b['x2'] for b in row_boxes), max(b['y2'] for b in row_boxes))

def calculate_ioa(row1, row2):
    r1_x1, r1_y1, r1_x2, r1_y2 = calculate_row_bbox(row1)
    r2_x1, r2_y1, r2_x2, r2_y2 = calculate_row_bbox(row2)
    
    area1 = max(0, r1_x2 - r1_x1) * max(0, r1_y2 - r1_y1)
    area2 = max(0, r2_x2 - r2_x1) * max(0, r2_y2 - r2_y1)
    
    if area1 == 0 or area2 == 0: return 0, 0
        
    ix1, iy1 = max(r1_x1, r2_x1), max(r1_y1, r2_y1)
    ix2, iy2 = min(r1_x2, r2_x2), min(r1_y2, r2_y2)
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    
    return inter_area / area1, inter_area / area2

def reconstruct_transcript(results):
    if not results: return ""
        
    for i, box in enumerate(results):
        box['id'] = i
        box['cx'] = (box['x1'] + box['x2']) / 2.0
        box['cy'] = (box['y1'] + box['y2']) / 2.0
        # The paper defines tolerance r(Bj) as half the diagonal of Bj [cite: 247]
        box['r'] = 0.5 * math.hypot(box['x2'] - box['x1'], box['y2'] - box['y1'])
        
    for i, box_i in enumerate(results):
        left_neighbors, right_neighbors = [], []
        for j, box_j in enumerate(results):
            if i == j: continue
            # Condition: Center of Bj is within r(Bj) of the horizontal line passing through Bi [cite: 247]
            if abs(box_j['cy'] - box_i['cy']) <= box_j['r']:
                if box_j['cx'] < box_i['cx']: left_neighbors.append(box_j)
                elif box_j['cx'] > box_i['cx']: right_neighbors.append(box_j)
                    
        box_i['left'] = max(left_neighbors, key=lambda b: b['cx'])['id'] if left_neighbors else None
        box_i['right'] = min(right_neighbors, key=lambda b: b['cx'])['id'] if right_neighbors else None

    unassociated = set(range(len(results)))
    candidate_rows = []
    # Lines are constructed greedily by selecting a previously unassociated box as a seed [cite: 248]
    while unassociated:
        seed_id = unassociated.pop()
        current_row = [seed_id]
        
        curr_id = seed_id
        while True:
            left_id = results[curr_id]['left']
            if left_id is not None and left_id in unassociated:
                current_row.insert(0, left_id)
                unassociated.remove(left_id)
                curr_id = left_id
            else: break
                
        curr_id = seed_id
        while True:
            right_id = results[curr_id]['right']
            if right_id is not None and right_id in unassociated:
                current_row.append(right_id)
                unassociated.remove(right_id)
                curr_id = right_id
            else: break
                
        candidate_rows.append([results[i] for i in current_row])
        
    merged = True
    while merged:
        merged = False
        new_rows = []
        skip = set()
        
        for i in range(len(candidate_rows)):
            if i in skip: continue
            merged_row = candidate_rows[i]
            for j in range(i + 1, len(candidate_rows)):
                if j in skip: continue
                ioa1, ioa2 = calculate_ioa(merged_row, candidate_rows[j])
                # Two rows with mutual IoA greater than 50% are merged, as are rows with a one-way IoA greater than 80% [cite: 250]
                if (ioa1 > 0.5 and ioa2 > 0.5) or ioa1 > 0.8 or ioa2 > 0.8:
                    merged_row.extend(candidate_rows[j])
                    skip.add(j)
                    merged = True
            new_rows.append(merged_row)
        candidate_rows = new_rows

    for row in candidate_rows: row.sort(key=lambda b: b['cx']) 
    candidate_rows.sort(key=lambda row: sum(b['cy'] for b in row) / len(row)) 
    
    return "\n".join("".join([b['label'] for b in row]) for row in candidate_rows)


# ==========================================
# 4. VISUALIZATION UTILITY
# ==========================================
def save_visualization(image_path, boxes, output_path):
    print(f"Generating visualization at {output_path}...")
    img = cv2.imread(image_path)
    
    for b in boxes:
        x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
        label = b['label']
        conf = b['conf']
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare text label (e.g., "Alpha 0.95")
        text = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text dimensions for the background box
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw a solid background rectangle for the text to ensure it's readable over the stone texture
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Draw the text in black over the green background
        cv2.putText(img, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
    cv2.imwrite(output_path, img)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greek Squeeze Pipeline")
    parser.add_argument("--img1", required=True, help="Path to Rotation 1 image")
    parser.add_argument("--img2", required=False, default=None, help="Path to Rotation 2 image (optional)")
    parser.add_argument("--out", required=True, help="Path to save the output .txt file")
    parser.add_argument("--vis_out", required=False, help="Optional: Path to save the visualized output image (.png/.jpg)")
    args = parser.parse_args()

    # Load resources
    detector, classifier, transforms, class_names = load_pipeline()
    
    # Run inferences (with logit averaging)
    boxes = run_tournament(args.img1, args.img2, detector, classifier, transforms, class_names)
    
    # Reconstruct text
    transcript = reconstruct_transcript(boxes)

    out_dir = os.path.dirname(args.out)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    
    # Save text to file
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(transcript)
        
    print(f"Pipeline complete. Transcript saved to {args.out}")
    
    # Generate visualization if requested
    if args.vis_out:
        save_visualization(args.img1, boxes, args.vis_out)
        print(f"Visualized image saved to {args.vis_out}")