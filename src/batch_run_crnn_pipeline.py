import os
import cv2
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ==========================================
# 1. CONFIGURATION (Set your paths here!)
# ==========================================
IMG_DIR = "../data/raw/Images"                               # Folder with images to process
OUT_DIR = "../outputs/crnn/batch_transcripts"                     # Folder to save .txt files
VIS_DIR = "../outputs/crnn/batch_visualizations"                  # Folder to save debug images (Set to None to disable)

YOLO_WEIGHTS = "../runs/obb/train3/weights/best.pt"           # Your trained YOLO model
CRNN_WEIGHTS = "../models/crnn_greek_20260420_155707_best.pth" # Your trained CRNN model
VOCAB_FILE   = "../data/processed/crnn_training_data/labels.txt"       # File used to build character map

# ==========================================
# 2. CRNN MODEL ARCHITECTURE
# ==========================================
class GreekCRNN(nn.Module):
    def __init__(self, img_channels=1, num_chars=35, hidden_size=256):
        super(GreekCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU(inplace=True) 
        )
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)

    def forward(self, x):
        conv_out = self.cnn(x).squeeze(2).permute(0, 2, 1) 
        rnn_out, _ = self.rnn(conv_out)
        return self.fc(rnn_out)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def decode_predictions(outputs, idx2char):
    _, max_indices = torch.max(outputs, dim=2)
    max_indices = max_indices.cpu().numpy()
    decoded_strings = []
    for sequence in max_indices:
        decoded_str = ""
        prev_char_idx = -1
        for char_idx in sequence:
            if char_idx != 0 and char_idx != prev_char_idx:
                decoded_str += idx2char[char_idx]
            prev_char_idx = char_idx
        decoded_strings.append(decoded_str)
    return decoded_strings

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def build_vocab(labels_file):
    unique_chars = set()
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                unique_chars.update(list(parts[1]))
    chars = sorted(list(unique_chars))
    char2idx = {char: idx + 1 for idx, char in enumerate(chars)}
    idx2char = {idx + 1: char for idx, char in enumerate(chars)}
    return char2idx, idx2char, len(chars)

# ==========================================
# 4. SINGLE IMAGE PROCESSING LOGIC
# ==========================================
def process_single_image(img_path, detection_model, crnn_model, char2idx, idx2char, device):
    img = cv2.imread(img_path)
    if img is None: raise ValueError("Could not read image file.")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. YOLO Inference
    res = get_sliced_prediction(
        img_path, detection_model, 
        slice_height=640, slice_width=640, 
        overlap_height_ratio=0.2, overlap_width_ratio=0.2, 
        postprocess_type="NMS", verbose=0
    )
    predictions = res.object_prediction_list
    if not predictions: return [], [], img

    # 2. Parse OBBs
    boxes_data = []
    for i, p in enumerate(predictions):
        if hasattr(p, 'mask') and p.mask is not None and hasattr(p.mask, 'polygon'):
            pts = np.array(p.mask.polygon[0]).reshape(-1, 2).astype("float32")
        else:
            x1, y1, x2, y2 = map(float, p.bbox.to_xyxy())
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")

        cx, cy = np.mean(pts, axis=0)
        diag1 = np.linalg.norm(pts[0] - pts[2]) if len(pts) == 4 else 0
        diag2 = np.linalg.norm(pts[1] - pts[3]) if len(pts) == 4 else 0
        r = max(diag1, diag2) * 0.5
        if r == 0: r = 20
        boxes_data.append({'id': i, 'pts': pts, 'cx': cx, 'cy': cy, 'r': r})

    # 3. Greedy Chaining (Slanted Baseline)
    for i, box_i in enumerate(boxes_data):
        left_neighbors, right_neighbors = [], []
        for j, box_j in enumerate(boxes_data):
            if i == j: continue
            if abs(box_j['cy'] - box_i['cy']) <= box_j['r']:
                if box_j['cx'] < box_i['cx']: left_neighbors.append(box_j)
                elif box_j['cx'] > box_i['cx']: right_neighbors.append(box_j)
                    
        box_i['left'] = max(left_neighbors, key=lambda b: b['cx'])['id'] if left_neighbors else None
        box_i['right'] = min(right_neighbors, key=lambda b: b['cx'])['id'] if right_neighbors else None

    unassociated = set(range(len(boxes_data)))
    candidate_rows = []

    while unassociated:
        seed_id = unassociated.pop()
        current_row = [seed_id]
        
        curr_id = seed_id
        while True:
            left_id = boxes_data[curr_id]['left']
            if left_id is not None and left_id in unassociated:
                current_row.insert(0, left_id)
                unassociated.remove(left_id)
                curr_id = left_id
            else: break
                
        curr_id = seed_id
        while True:
            right_id = boxes_data[curr_id]['right']
            if right_id is not None and right_id in unassociated:
                current_row.append(right_id)
                unassociated.remove(right_id)
                curr_id = right_id
            else: break
                
        candidate_rows.append([boxes_data[i] for i in current_row])

    lines = []
    for row in candidate_rows:
        row.sort(key=lambda b: b['cx'])
        lines.append(row)
    lines.sort(key=lambda row: sum(b['cy'] for b in row) / len(row))

    # 4. CRNN Warping & Recognition
    final_transcript = []
    for line in lines:
        all_pts = np.vstack([b['pts'] for b in line])
        rect = cv2.minAreaRect(all_pts)
        (cx, cy), (w, h), angle = rect
        if h > w: w, h, angle = h, w, angle - 90.0
            
        padded_w = w + h 
        padded_rect = ((cx, cy), (padded_w, h), angle)
        box = cv2.boxPoints(padded_rect)
        src_pts = order_points(box)
        
        target_h = 32
        target_w = max(1, int(padded_w * (target_h / max(1, h))))
        dst_pts = np.array([[0,0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        line_crop = cv2.warpPerspective(img_gray, M, (target_w, target_h), borderValue=0)
        
        input_tensor = torch.FloatTensor((line_crop / 127.5) - 1.0).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = crnn_model(input_tensor)
            pred_text = decode_predictions(output, idx2char)[0]
            
        final_transcript.append(pred_text)
        
        # FIX: Removed the bad assignment that broke the script
        # line['pred_text'] = pred_text 

    return final_transcript, lines, img

# ==========================================
# 5. MAIN BATCH PROCESS
# ==========================================
def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"--- BATCH INITIALIZATION ({device}) ---")

    # 1. Setup Directories & Diagnostics
    os.makedirs(OUT_DIR, exist_ok=True)
    if VIS_DIR: os.makedirs(VIS_DIR, exist_ok=True)
    
    diag_file = os.path.join(OUT_DIR, "batch_diagnostics.tsv")
    
    # 2. Load Models Once
    char2idx, idx2char, num_chars = build_vocab(VOCAB_FILE)
    
    print("Loading YOLO model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', model_path=YOLO_WEIGHTS,
        confidence_threshold=0.25, device=device
    )
    
    print("Loading CRNN model...")
    crnn_model = GreekCRNN(num_chars=num_chars).to(device)
    crnn_model.load_state_dict(torch.load(CRNN_WEIGHTS, map_location=device))
    crnn_model.eval()

    # 3. Find Images
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(image_paths)} images to process.\n" + "-"*40)

    # 4. Batch Loop
    diagnostics = []
    
    for idx, img_path in enumerate(image_paths):
        base_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(base_name)[0]
        start_time = time.time()
        
        try:
            # Run Inference
            transcript, lines_data, original_img = process_single_image(
                img_path, detection_model, crnn_model, char2idx, idx2char, device
            )
            
            # Save Transcript
            out_txt_path = os.path.join(OUT_DIR, f"{name_no_ext}.txt")
            with open(out_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(transcript))
                
            # Save Visualization (If requested)
            if VIS_DIR and len(lines_data) > 0:
                vis_img = original_img.copy()
                np.random.seed(42)
                for i, line in enumerate(lines_data):
                    color = tuple(np.random.randint(50, 255, 3).tolist())
                    
                    # FIX: Safely pull the text from the transcript array instead of the list
                    current_text = transcript[i] 
                    
                    for j in range(len(line) - 1):
                        cv2.line(vis_img, (int(line[j]['cx']), int(line[j]['cy'])), 
                                 (int(line[j+1]['cx']), int(line[j+1]['cy'])), color, 3)
                    for box in line:
                        cv2.polylines(vis_img, [np.int32(box['pts'])], True, color, 2)
                        
                    cv2.putText(vis_img, f"L{i+1}: {current_text}", 
                                (int(line[0]['cx']) - 50, int(line[0]['cy']) - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                vis_path = os.path.join(VIS_DIR, f"{name_no_ext}_vis.jpg")
                cv2.imwrite(vis_path, vis_img)

            # Record Success
            elapsed = time.time() - start_time
            total_chars = sum(len(t) for t in transcript)
            diagnostics.append(f"{base_name}\tSUCCESS\t{elapsed:.2f}\t{len(lines_data)}\t{total_chars}\tNone")
            print(f"[{idx+1}/{len(image_paths)}] ✅ Processed {base_name} ({len(lines_data)} lines) in {elapsed:.2f}s")
            
        except Exception as e:
            # Record Failure but KEEP GOING
            elapsed = time.time() - start_time
            diagnostics.append(f"{base_name}\tERROR\t{elapsed:.2f}\t0\t0\t{str(e)}")
            print(f"[{idx+1}/{len(image_paths)}] ❌ Error on {base_name}: {e}")

    # 5. Save Diagnostics File
    with open(diag_file, 'w', encoding='utf-8') as f:
        f.write("Image_Name\tStatus\tTime_Sec\tNum_Lines\tTotal_Chars\tError_Msg\n")
        f.write("\n".join(diagnostics))
        
    print("\n" + "="*40)
    print(f"BATCH COMPLETE! Diagnostics saved to: {diag_file}")
    print("="*40)

if __name__ == "__main__":
    main()