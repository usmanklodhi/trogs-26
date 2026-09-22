import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import shutil
import time
from PIL import Image
from torchvision import transforms, models
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ==========================================
# 1. SETUP AND CONFIGURATION
# ==========================================
YOLO_MODEL_PATH = './runs/obb/train3/weights/best.pt'
CLS_MODEL_PATH = './models/greek_classifier_resnet18_20260214_212940.pth'
TRAIN_DIR = './data/processed/greek_letters_classification/train'
PROXIMITY_THRESHOLD = 50 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pipeline():
    print(f"Initializing models on {DEVICE}...")
    CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=YOLO_MODEL_PATH,
        confidence_threshold=0.50,
        device=DEVICE,
    )

    classifier = models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, len(CLASS_NAMES))
    classifier.load_state_dict(torch.load(CLS_MODEL_PATH, map_location='cpu'))
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
# def run_tournament(img1_path, img2_path, detector, classifier, transforms, class_names):
#     print("Running detection on Image 1...")
#     res1 = get_sliced_prediction(img1_path, detector, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, postprocess_type="NMS")
#     boxes1 = res1.object_prediction_list

#     img1_cv2 = cv2.imread(img1_path)
#     h1, w1, _ = img1_cv2.shape

#     if img2_path is None:
#         print("No Image 2 provided, processing single image...")
#         boxes2 = []
#         img2_cv2 = None
#         h2 = w2 = 0
#     else:
#         print("Running detection on Image 2...")
#         res2 = get_sliced_prediction(img2_path, detector, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, postprocess_type="NMS")
#         boxes2 = res2.object_prediction_list

#         img2_cv2 = cv2.imread(img2_path)
#         h2, w2, _ = img2_cv2.shape

#     final_results = []
    
#     print("Running Proximity Matching & Logit Averaging Fusion...")
#     for b1 in boxes1:
#         x1_min, y1_min, x1_max, y1_max = map(int, b1.bbox.to_xyxy())
#         cx1, cy1 = (x1_min + x1_max) / 2, (y1_min + y1_max) / 2
        
#         closest_b2 = None
#         min_dist = float('inf')
        
#         for b2 in boxes2:
#             x2_min, y2_min, x2_max, y2_max = map(int, b2.bbox.to_xyxy())
#             cx2, cy2 = (x2_min + x2_max) / 2, (y2_min + y2_max) / 2
#             dist = math.hypot(cx1 - cx2, cy1 - cy2)
            
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_b2 = b2
#                 best_coords2 = (x2_min, y2_min, x2_max, y2_max)
                
#         if closest_b2 and min_dist < PROXIMITY_THRESHOLD:
#             x2_min, y2_min, x2_max, y2_max = best_coords2
            
#             # Safe Crop 1
#             x1_min_c, y1_min_c = max(0, x1_min), max(0, y1_min)
#             x1_max_c, y1_max_c = min(w1, x1_max), min(h1, y1_max)
#             crop1 = img1_cv2[y1_min_c:y1_max_c, x1_min_c:x1_max_c]
            
#             # Safe Crop 2
#             x2_min_c, y2_min_c = max(0, x2_min), max(0, y2_min)
#             x2_max_c, y2_max_c = min(w2, x2_max), min(h2, y2_max)
#             crop2 = img2_cv2[y2_min_c:y2_max_c, x2_min_c:x2_max_c]
            
#             if crop1.size == 0 or crop2.size == 0: continue
            
#             t1 = transforms(Image.fromarray(cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
#             t2 = transforms(Image.fromarray(cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
            
#             with torch.no_grad():
#                 out1, out2 = classifier(t1), classifier(t2)
                
#                 # --- LOGIT AVERAGING (LATE FUSION) ---
#                 avg_logits = (out1 + out2) / 2.0
#                 avg_probs = F.softmax(avg_logits, dim=1)
#                 best_conf_tensor, best_p_tensor = torch.max(avg_probs, 1)
                
#                 best_label = class_names[best_p_tensor.item()]
#                 best_conf = best_conf_tensor.item()
                
#             final_results.append({
#                 'x1': x1_min, 'y1': y1_min, 'x2': x1_max, 'y2': y1_max, 
#                 'label': best_label, 'conf': best_conf
#             })

#         else:
#             x1_min_c, y1_min_c = max(0, x1_min), max(0, y1_min)
#             x1_max_c, y1_max_c = min(w1, x1_max), min(h1, y1_max)
#             crop1 = img1_cv2[y1_min_c:y1_max_c, x1_min_c:x1_max_c]
#             if crop1.size == 0: continue
            
#             t1 = transforms(Image.fromarray(cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
#             with torch.no_grad():
#                 conf1, p1 = torch.max(F.softmax(classifier(t1), dim=1), 1)
                
#             final_results.append({
#                 'x1': x1_min, 'y1': y1_min, 'x2': x1_max, 'y2': y1_max, 
#                 'label': class_names[p1.item()], 'conf': conf1.item()
#             })
            
#     return final_results

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
        det_conf1 = b1.score.value # Capture YOLO detection confidence
        
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
            det_conf2 = closest_b2.score.value # Capture secondary YOLO detection confidence
            
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
                avg_det_conf = (det_conf1 + det_conf2) / 2.0 # Average the detection confidences
                
            final_results.append({
                'x1': x1_min, 'y1': y1_min, 'x2': x1_max, 'y2': y1_max, 
                'label': best_label, 
                'conf': best_conf,
                'det_conf': avg_det_conf
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
                'label': class_names[p1.item()], 
                'conf': conf1.item(),
                'det_conf': det_conf1
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


# # ==========================================
# # 3. VISUAL MORPHOLOGY RECONSTRUCTION (ELLIPTICAL SMEAR)
# # ==========================================
# def reconstruct_transcript(results):
#     if not results: return ""
    
#     # 1. Add center points and determine dynamic canvas size
#     max_x, max_y = 0, 0
#     avg_h, avg_w = 0, 0
    
#     for i, box in enumerate(results):
#         box['cx'] = (box['x1'] + box['x2']) / 2.0
#         box['cy'] = (box['y1'] + box['y2']) / 2.0
        
#         h = box['y2'] - box['y1']
#         w = box['x2'] - box['x1']
#         avg_h += h
#         avg_w += w
        
#         if box['x2'] > max_x: max_x = box['x2']
#         if box['y2'] > max_y: max_y = box['y2']
        
#     avg_h /= len(results)
#     avg_w /= len(results)
    
#     # 2. Create a blank canvas just large enough to hold all boxes
#     # (Adding 100px padding so edge letters don't get clipped during dilation)
#     canvas_w, canvas_h = int(max_x + 100), int(max_y + 100)
#     mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
#     # 3. Paint every YOLO box solid white on the canvas
#     for box in results:
#         cv2.rectangle(mask, (int(box['x1']), int(box['y1'])), (int(box['x2']), int(box['y2'])), 255, -1)
        
#     # 4. HORIZONTAL ELLIPTICAL SMEAR (The Magic Trick for Tilted Text)
#     # Width: 3.5x char width to bridge gaps between words
#     # Height: 20% char height to prevent vertical bleeding
#     kernel_width = max(3, int(avg_w * 3.5))  
#     kernel_height = max(1, int(avg_h * 0.2))   
    
#     # OpenCV prefers odd-numbered kernel sizes for a perfect center anchor
#     if kernel_width % 2 == 0: kernel_width += 1
#     if kernel_height % 2 == 0: kernel_height += 1
        
#     # THE FIX: Using an Ellipse to smoothly bridge diagonal gaps on sloping text!
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))
    
#     smeared_mask = cv2.dilate(mask, kernel, iterations=1)
    
#     # 5. Extract the fused text lines
#     contours, _ = cv2.findContours(smeared_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # 6. Group the original YOLO boxes by which contour they fall inside
#     candidate_rows = []
#     for contour in contours:
#         line_boxes = []
        
#         # Create a temporary mask just for this specific text line
#         line_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
#         cv2.drawContours(line_mask, [contour], -1, 255, -1)
        
#         for box in results:
#             cx, cy = int(box['cx']), int(box['cy'])
#             # If the center of the character is inside the painted line, it belongs to this row
#             if line_mask[cy, cx] == 255:
#                 line_boxes.append(box)
                
#         if line_boxes:
#             candidate_rows.append(line_boxes)
            
#     # 7. Sort Rows top-to-bottom, and Characters left-to-right
#     for row in candidate_rows: 
#         row.sort(key=lambda b: b['cx']) 
        
#     # Sort rows top-to-bottom based on their average Y coordinate
#     candidate_rows.sort(key=lambda row: sum(b['cy'] for b in row) / len(row)) 
    
#     # 8. Generate the final transcript string
#     return "\n".join("".join([b['label'] for b in row]) for row in candidate_rows)


# ==========================================
# 4. VISUALIZATION UTILITY
# ==========================================
# def save_visualization(image_path, boxes, output_path):
#     print(f"Generating visualization at {output_path}...")
#     img = cv2.imread(image_path)
    
#     for b in boxes:
#         x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
#         label = b['label']
#         conf = b['conf']
        
#         # Draw bounding box
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
#         # Prepare text label (e.g., "Alpha 0.95")
#         text = f"{label} {conf:.2f}"
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.5
#         thickness = 1
        
#         # Get text dimensions for the background box
#         (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
#         # Draw a solid background rectangle for the text to ensure it's readable over the stone texture
#         cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        
#         # Draw the text in black over the green background
#         cv2.putText(img, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
#     cv2.imwrite(output_path, img)

def save_visualization(image_path, boxes, output_path):
    print(f"Generating visualization at {output_path}...")
    img = cv2.imread(image_path)
    
    for b in boxes:
        x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
        label = b['label']
        cls_conf = b['conf']
        det_conf = b.get('det_conf', 0.0) # safely get detection conf
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare text label (e.g., "Alpha [C:0.95 D:0.88]")
        text = f"{label} [C:{cls_conf:.2f} D:{det_conf:.2f}]"
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
# 5. MAIN EXECUTION (BATCH PROCESSING WITH MODELS LOADED ONCE)
# ==========================================
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = f'./results/batch_run_{timestamp}'
    OUT_DIR = f'{RUN_DIR}/transcripts'
    VIS_DIR = f'{RUN_DIR}/visualizations'
    DIAG_FILE = f'{RUN_DIR}/diagnostics.txt'

    RAW_DIR = './data/raw/Images' # this i have to change to the directory where the images are stored

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # Load models once at the beginning
    detector, classifier, transforms, class_names = load_pipeline()

    # List all files ending with _Rotation1_300dpi.png or _Rotation1_300dpi.tif
    all_files = os.listdir(RAW_DIR)
    rotation1_files = [f for f in all_files if f.endswith(('_Rotation1_300dpi.png', '_Rotation1_300dpi.tif'))]

    diagnostics = []

    for file1 in rotation1_files:
        if file1.endswith('_Rotation1_300dpi.png'):
            base = file1.replace('_Rotation1_300dpi.png', '')
            ext = 'png'
        elif file1.endswith('_Rotation1_300dpi.tif'):
            base = file1.replace('_Rotation1_300dpi.tif', '')
            ext = 'tif'
        else:
            continue  # Skip if doesn't match
        
        file2_png = f'{base}_Rotation2_300dpi.png'
        file2_tif = f'{base}_Rotation2_300dpi.tif'
        if os.path.exists(os.path.join(RAW_DIR, file2_png)):
            file2 = file2_png
        elif os.path.exists(os.path.join(RAW_DIR, file2_tif)):
            file2 = file2_tif
        else:
            file2 = None
        
        path1 = os.path.join(RAW_DIR, file1)
        path2 = os.path.join(RAW_DIR, file2) if file2 else None
        has_rotation2 = file2 is not None
        
        out_txt1 = os.path.join(OUT_DIR, f'{base}_Rotation1_300dpi_transcript.txt')
        if has_rotation2:
            out_txt2 = os.path.join(OUT_DIR, f'{base}_Rotation2_300dpi_transcript.txt')
        out_vis = os.path.join(VIS_DIR, f'{base}_Visualized.png')
        
        print(f"Processing {base}...")
        start = time.time()
        try:
            # Run the tournament directly without subprocess
            boxes = run_tournament(path1, path2 if has_rotation2 else None, detector, classifier, transforms, class_names)
            
            # Reconstruct text
            transcript = reconstruct_transcript(boxes)
            
            # Compute additional diagnostics
            num_boxes = len(boxes)
            transcript_length = len(transcript)
            num_rows = len(transcript.split('\n')) if transcript else 0
            avg_conf = sum(b['conf'] for b in boxes) / len(boxes) if boxes else 0.0
            
            # Save text
            with open(out_txt1, 'w', encoding='utf-8') as f:
                f.write(transcript)
                
            if has_rotation2:
                # Copy the transcript to the second file
                shutil.copy(out_txt1, out_txt2)
                status = 'success'
                msg = f"Done {base} in {time.time() - start:.2f} seconds. Transcripts saved to {out_txt1} and {out_txt2}"
            else:
                status = 'success_single'
                msg = f"Done {base} (single image) in {time.time() - start:.2f} seconds. Transcript saved to {out_txt1}"
            
            # Save visualization
            save_visualization(path1, boxes, out_vis)
            
            diagnostics.append({
                'base': base,
                'img1': path1,
                'img2': path2,
                'status': status,
                'time_sec': time.time() - start,
                'error': None,
                'num_boxes': num_boxes,
                'transcript_length': transcript_length,
                'num_rows': num_rows,
                'avg_conf': avg_conf
            })
            print(msg)
        except Exception as e:
            elapsed = time.time() - start
            diagnostics.append({
                'base': base,
                'img1': path1,
                'img2': path2 if has_rotation2 else None,
                'status': 'error',
                'time_sec': elapsed,
                'error': str(e),
                'num_boxes': 0,
                'transcript_length': 0,
                'num_rows': 0,
                'avg_conf': 0.0
            })
            print(f"Error processing {base}: {e}")

    # Save diagnostics to file
    with open(DIAG_FILE, 'w', encoding='utf-8') as f:
        f.write('base\timg1\timg2\tstatus\ttime_sec\terror\tnum_boxes\ttranscript_length\tnum_rows\tavg_conf\n')
        for d in diagnostics:
            f.write(f"{d['base']}\t{d['img1']}\t{d['img2']}\t{d['status']}\t{d['time_sec']}\t{d['error']}\t{d['num_boxes']}\t{d['transcript_length']}\t{d['num_rows']}\t{d['avg_conf']}\n")

    print(f"Batch processing complete. Results saved to {RUN_DIR}. Diagnostics saved to {DIAG_FILE}")