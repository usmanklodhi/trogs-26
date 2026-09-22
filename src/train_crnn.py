# import os
# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import random

# import logging
# from datetime import datetime

# import Levenshtein

# # ==========================================
# # 1. DATASET & COLLATE FUNCTION
# # ==========================================
# class GreekSqueezeDataset(Dataset):
#     def __init__(self, entries, img_dir, char2idx):
#         self.entries = entries
#         self.img_dir = img_dir
#         self.char2idx = char2idx

#     def __len__(self):
#         return len(self.entries)

#     def __getitem__(self, idx):
#         filename, text = self.entries[idx]
#         img_path = os.path.join(self.img_dir, filename)
        
#         # Load grayscale
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             # Fallback for missing images to prevent crashes
#             img = np.zeros((32, 100), dtype=np.uint8) 
            
#         # Normalize to [-1, 1]
#         img = (img / 127.5) - 1.0
#         img_tensor = torch.FloatTensor(img).unsqueeze(0) 
        
#         target = [self.char2idx[c] for c in text]
#         return img_tensor, target

# def crnn_collate_fn(batch):
#     images, targets = zip(*batch)
#     max_width = max([img.shape[2] for img in images])
    
#     padded_images = []
#     for img in images:
#         c, h, w = img.shape
#         pad_amount = max_width - w
#         # Pad right side with -1.0 (black background)
#         padded_img = F.pad(img, (0, pad_amount, 0, 0), value=-1.0)
#         padded_images.append(padded_img)
        
#     images_tensor = torch.stack(padded_images)
    
#     target_lengths = torch.IntTensor([len(t) for t in targets])
#     targets_flat = torch.IntTensor([char_idx for t in targets for char_idx in t])
    
#     # We also return the raw target lists for decoding comparisons
#     return images_tensor, targets_flat, target_lengths, targets

# # ==========================================
# # 2. MODEL ARCHITECTURE
# # ==========================================
# class GreekCRNN(nn.Module):
#     def __init__(self, img_channels=1, num_chars=35, hidden_size=256):
#         super(GreekCRNN, self).__init__()
        
#         self.cnn = nn.Sequential(
#             nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
#             nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True) 
#         )
        
#         self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size, 
#                            num_layers=2, bidirectional=True, batch_first=True)
        
#         self.fc = nn.Linear(hidden_size * 2, num_chars + 1)

#     def forward(self, x):
#         conv_out = self.cnn(x)
#         conv_out = conv_out.squeeze(2).permute(0, 2, 1) 
#         rnn_out, _ = self.rnn(conv_out)
#         out = self.fc(rnn_out)
#         return out

# # ==========================================
# # 3. CTC DECODER (To watch it learn)
# # ==========================================
# def decode_predictions(outputs, idx2char):
#     """
#     Decodes the raw probability matrix into a readable string.
#     outputs shape: [Batch, Seq_Len, Num_Classes]
#     """
#     # 1. Get the argmax across the character classes
#     _, max_indices = torch.max(outputs, dim=2)
#     max_indices = max_indices.cpu().numpy()
    
#     decoded_strings = []
#     for sequence in max_indices:
#         decoded_str = ""
#         prev_char_idx = -1
        
#         for char_idx in sequence:
#             # CTC Logic: Ignore blanks (0) and consecutive duplicates
#             if char_idx != 0 and char_idx != prev_char_idx:
#                 decoded_str += idx2char[char_idx]
#             prev_char_idx = char_idx
            
#         decoded_strings.append(decoded_str)
        
#     return decoded_strings

# def evaluate_cer(model, val_loader, device, idx2char):
#     model.eval()
#     total_distance = 0
#     total_chars = 0
    
#     with torch.no_grad():
#         for images, _, _, targets_raw in val_loader:
#             images = images.to(device)
#             outputs = model(images)
            
#             # Decode the batch of predictions
#             predictions = decode_predictions(outputs, idx2char)
            
#             # Compare each prediction to its ground truth
#             for pred_str, true_raw in zip(predictions, targets_raw):
#                 true_str = "".join([idx2char[c] for c in true_raw])
                
#                 # Levenshtein distance handles the Insertions, Deletions, and Substitutions
#                 total_distance += Levenshtein.distance(pred_str, true_str)
#                 total_chars += len(true_str)
                
#     cer = total_distance / max(total_chars, 1) # Prevent division by zero
#     return cer

# # ==========================================
# # 4. TRAINING LOOP (With Logging & Timestamps)
# # ==========================================
# def main():
#     # --- SETUP LOGGING & TIMESTAMPS ---
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     os.makedirs("logs", exist_ok=True)
#     os.makedirs("models", exist_ok=True)
    
#     log_filename = os.path.join("logs", f"training_crnn_{timestamp}.log")
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_filename),
#             logging.StreamHandler() # Keeps printing to the console
#         ]
#     )
    
#     logging.info("=== Starting CRNN Training Pipeline ===")
#     logging.info(f"Log file created: {log_filename}")

#     # Setup Paths
#     DATA_DIR = "data/processed/crnn_training_data"
#     LABELS_FILE = os.path.join(DATA_DIR, "labels.txt")
#     IMG_DIR = os.path.join(DATA_DIR, "images")
    
#     # 1. Read Manifest & Build Vocabulary
#     data_entries = []
#     with open(LABELS_FILE, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             if len(parts) == 2:
#                 data_entries.append((parts[0], parts[1]))

#     unique_chars = set()
#     for _, text in data_entries:
#         unique_chars.update(list(text))

#     chars = sorted(list(unique_chars))
#     char2idx = {char: idx + 1 for idx, char in enumerate(chars)}
#     idx2char = {idx + 1: char for idx, char in enumerate(chars)}
    
#     logging.info(f"Vocabulary Size: {len(chars)} characters + 1 Blank")

#     # 2. Split Data & Create Loaders
#     random.shuffle(data_entries)
#     split_idx = int(len(data_entries) * 0.85)
    
#     train_data = GreekSqueezeDataset(data_entries[:split_idx], IMG_DIR, char2idx)
#     val_data = GreekSqueezeDataset(data_entries[split_idx:], IMG_DIR, char2idx)

#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=crnn_collate_fn, num_workers=4)
#     val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=crnn_collate_fn, num_workers=4)
    
#     # 3. Initialize Hardware, Model, Loss, Optimizer
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Training hardware: {device}")
    
#     model = GreekCRNN(num_chars=len(chars)).to(device)
#     criterion = nn.CTCLoss(blank=0, zero_infinity=True)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # 4. The Loop
#     EPOCHS = 30
#     best_loss = float('inf')
#     best_model_path = os.path.join("models", f"crnn_greek_{timestamp}_best.pth")

#     logging.info(f"Starting training for {EPOCHS} epochs...")

#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss = 0
        
#         for batch_idx, (images, targets_flat, target_lengths, targets_raw) in enumerate(train_loader):
#             images = images.to(device)
#             targets_flat = targets_flat.to(device)
#             target_lengths = target_lengths.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(images)
            
#             # Log_softmax and permute for CTCLoss
#             outputs_log = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
            
#             input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(device)
            
#             loss = criterion(outputs_log, targets_flat, input_lengths, target_lengths)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
            
#             # Watch the network learn every 50 batches
#             if batch_idx % 50 == 0:
#                 logging.info(f"Epoch {epoch+1:02d}/{EPOCHS} | Batch {batch_idx:03d}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
#                 # Decode the first image in the batch to check progress
#                 model.eval()
#                 with torch.no_grad():
#                     sample_output = model(images[:1])
#                     pred_str = decode_predictions(sample_output, idx2char)[0]
#                     true_str = "".join([idx2char[c] for c in targets_raw[0]])
                    
#                     logging.info(f"   Target:    {true_str}")
#                     logging.info(f"   Predicted: {pred_str}")
#                 model.train()

#         avg_loss = total_loss / len(train_loader)
#         logging.info(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---")
       
#         # Inside your main training loop, after the train_loader loop:
#         val_cer = evaluate_cer(model, val_loader, device, idx2char)
#         logging.info(f"Validation CER: {val_cer * 100:.2f}%")
        
#         # Save best model
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save(model.state_dict(), best_model_path)
#             logging.info(f">>> New best loss! Model saved to: {best_model_path}")

#     logging.info("=== Training Complete ===")

# if __name__ == "__main__":
#     main()

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import logging
from datetime import datetime

import Levenshtein

# ==========================================
# 1. DATASET & COLLATE FUNCTION
# ==========================================
class GreekSqueezeDataset(Dataset):
    def __init__(self, entries, img_dir, char2idx):
        self.entries = entries
        self.img_dir = img_dir
        self.char2idx = char2idx

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        filename, text = self.entries[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        # Load grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for missing images to prevent crashes
            img = np.zeros((32, 100), dtype=np.uint8) 
            
        # Normalize to [-1, 1]
        img = (img / 127.5) - 1.0
        img_tensor = torch.FloatTensor(img).unsqueeze(0) 
        
        target = [self.char2idx[c] for c in text]
        return img_tensor, target

def crnn_collate_fn(batch):
    images, targets = zip(*batch)
    max_width = max([img.shape[2] for img in images])
    
    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_amount = max_width - w
        # Pad right side with -1.0 (black background)
        padded_img = F.pad(img, (0, pad_amount, 0, 0), value=-1.0)
        padded_images.append(padded_img)
        
    images_tensor = torch.stack(padded_images)
    
    target_lengths = torch.IntTensor([len(t) for t in targets])
    targets_flat = torch.IntTensor([char_idx for t in targets for char_idx in t])
    
    # We also return the raw target lists for decoding comparisons
    return images_tensor, targets_flat, target_lengths, targets

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class GreekCRNN(nn.Module):
    def __init__(self, img_channels=1, num_chars=35, hidden_size=256):
        super(GreekCRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True) 
        )
        
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size, 
                           num_layers=2, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)

    def forward(self, x):
        conv_out = self.cnn(x)
        conv_out = conv_out.squeeze(2).permute(0, 2, 1) 
        rnn_out, _ = self.rnn(conv_out)
        out = self.fc(rnn_out)
        return out

# ==========================================
# 3. CTC DECODER & EVALUATION
# ==========================================
def decode_predictions(outputs, idx2char):
    """
    Decodes the raw probability matrix into a readable string.
    outputs shape: [Batch, Seq_Len, Num_Classes]
    """
    _, max_indices = torch.max(outputs, dim=2)
    max_indices = max_indices.cpu().numpy()
    
    decoded_strings = []
    for sequence in max_indices:
        decoded_str = ""
        prev_char_idx = -1
        
        for char_idx in sequence:
            # CTC Logic: Ignore blanks (0) and consecutive duplicates
            if char_idx != 0 and char_idx != prev_char_idx:
                decoded_str += idx2char[char_idx]
            prev_char_idx = char_idx
            
        decoded_strings.append(decoded_str)
        
    return decoded_strings

def evaluate_metrics(model, val_loader, criterion, device, idx2char):
    """Calculates both Validation Loss and Character Error Rate (CER)"""
    model.eval()
    total_distance = 0
    total_chars = 0
    total_val_loss = 0
    
    with torch.no_grad():
        for images, targets_flat, target_lengths, targets_raw in val_loader:
            images = images.to(device)
            targets_flat = targets_flat.to(device)
            target_lengths = target_lengths.to(device)
            
            outputs = model(images)
            
            # Calculate Validation Loss
            outputs_log = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(device)
            loss = criterion(outputs_log, targets_flat, input_lengths, target_lengths)
            total_val_loss += loss.item()
            
            # Calculate CER
            predictions = decode_predictions(outputs, idx2char)
            for pred_str, true_raw in zip(predictions, targets_raw):
                true_str = "".join([idx2char[c] for c in true_raw])
                total_distance += Levenshtein.distance(pred_str, true_str)
                total_chars += len(true_str)
                
    avg_val_loss = total_val_loss / len(val_loader)
    cer = total_distance / max(total_chars, 1) # Prevent division by zero
    
    return avg_val_loss, cer

# ==========================================
# 4. TRAINING LOOP & THESIS LOGGING
# ==========================================
def main():
    # --- SETUP LOGGING & TIMESTAMPS ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    log_filename = os.path.join("logs", f"training_crnn_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=== Starting CRNN Training Pipeline ===")

    # Setup Paths
    DATA_DIR = "data/processed/crnn_training_data"
    LABELS_FILE = os.path.join(DATA_DIR, "labels.txt")
    IMG_DIR = os.path.join(DATA_DIR, "images")
    
    # 1. Read Manifest & Build Vocabulary
    data_entries = []
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                data_entries.append((parts[0], parts[1]))

    unique_chars = set()
    for _, text in data_entries:
        unique_chars.update(list(text))

    chars = sorted(list(unique_chars))
    char2idx = {char: idx + 1 for idx, char in enumerate(chars)}
    idx2char = {idx + 1: char for idx, char in enumerate(chars)}
    
    # 2. Split Data & Create Loaders
    random.shuffle(data_entries)
    split_idx = int(len(data_entries) * 0.85)
    
    train_data = GreekSqueezeDataset(data_entries[:split_idx], IMG_DIR, char2idx)
    val_data = GreekSqueezeDataset(data_entries[split_idx:], IMG_DIR, char2idx)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=crnn_collate_fn, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=crnn_collate_fn, num_workers=4)
    
    # 3. Initialize Hardware, Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GreekCRNN(num_chars=len(chars)).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. The Loop
    EPOCHS = 30
    best_val_loss = float('inf')
    best_model_path = os.path.join("models", f"crnn_greek_{timestamp}_best.pth")

    # Data structures to hold metrics for exporting
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_cer": []
    }

    logging.info(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, targets_flat, target_lengths, targets_raw) in enumerate(train_loader):
            images = images.to(device)
            targets_flat = targets_flat.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            outputs_log = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(device)
            
            loss = criterion(outputs_log, targets_flat, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logging.info(f"Epoch {epoch+1:02d}/{EPOCHS} | Batch {batch_idx:03d}/{len(train_loader)} | Train Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Phase
        val_loss, val_cer = evaluate_metrics(model, val_loader, criterion, device, idx2char)
        
        logging.info(f"--- Epoch {epoch+1} Completed ---")
        logging.info(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {val_loss:.4f} | Val CER: {val_cer * 100:.2f}%")
        
        # Save metrics for plotting/Excel
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_cer"].append(val_cer)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f">>> New best val loss! Model saved.")

    logging.info("=== Training Complete ===")

    # ==========================================
    # 5. GENERATE THESIS ASSETS (CSV, Plots, Text)
    # ==========================================
    logging.info("Generating assets for thesis writing...")
    
    # A. Save to CSV (Excel compatible)
    df = pd.DataFrame(history)
    csv_path = os.path.join("logs", f"metrics_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # B. Generate and save Loss Curves plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', marker='o', markersize=4)
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red', marker='x', markersize=4)
    plt.title('CRNN + CTC Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('CTC Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join("logs", f"loss_curves_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # C. Generate Thesis Text Summary
    final_cer = history['val_cer'][-1] * 100
    best_cer = min(history['val_cer']) * 100
    
    # Basic analysis for spikes (checks if val loss standard deviation is unusually high or if it went back up significantly)
    val_loss_diffs = np.diff(history['val_loss'])
    spikes_detected = any(diff > (sum(history['val_loss']) / len(history['val_loss'])) * 0.2 for diff in val_loss_diffs)
    
    curve_behavior = "smoothly decreased and plateaued"
    if spikes_detected:
         curve_behavior = "experienced some volatility or spikes before settling"
    elif history['val_loss'][-1] > best_val_loss * 1.15:
         curve_behavior = "decreased initially but showed signs of overfitting (spiking) towards the end"

    thesis_summary_path = os.path.join("logs", f"thesis_summary_partB_{timestamp}.txt")
    with open(thesis_summary_path, "w") as f:
        f.write("====================================================\n")
        f.write("Part B: Pathway 2 Results (CRNN + CTC)\n")
        f.write("====================================================\n\n")
        f.write(f"Final CER:\n")
        f.write(f"- The final Character Error Rate (CER) on the validation set was {final_cer:.2f}% at epoch {EPOCHS}.\n")
        f.write(f"- Note: The absolute best CER achieved during training was {best_cer:.2f}%.\n\n")
        f.write(f"Loss Curves:\n")
        f.write(f"- Based on the saved metrics, the validation loss {curve_behavior}.\n")
        f.write(f"- Initial Validation Loss: {history['val_loss'][0]:.4f}\n")
        f.write(f"- Final Validation Loss: {history['val_loss'][-1]:.4f}\n\n")
        f.write("Files generated for your thesis:\n")
        f.write(f"1. Graph: {plot_path}\n")
        f.write(f"2. Data (Excel/CSV): {csv_path}\n")
        
    logging.info(f"Assets generated successfully. Check the 'logs' folder.")

if __name__ == "__main__":
    main()