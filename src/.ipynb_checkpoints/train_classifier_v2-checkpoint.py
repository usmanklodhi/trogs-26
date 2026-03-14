import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
import copy
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "data/processed/greek_letters_classification"

# Generate a unique timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = f"models/greek_classifier_resnet18_{timestamp}.pth"
CM_SAVE_PATH = f"models/confusion_matrix_{timestamp}.png"

BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Model will be saved to: {MODEL_SAVE_PATH}")

# # --- CUSTOM DATASET CLASS (The Fix) ---
# class FixedLabelsImageFolder(datasets.ImageFolder):
#     """
#     Overrides ImageFolder to force a specific class-to-index mapping.
#     This ensures Train and Val use the exact same indices, even if Val is missing folders.
#     """
#     def __init__(self, root, transform=None, class_to_idx=None):
#         self.forced_class_to_idx = class_to_idx
#         super().__init__(root, transform=transform)

#     def find_classes(self, directory):
#         if self.forced_class_to_idx is not None:
#             # Force the mapping we provided
#             classes = list(self.forced_class_to_idx.keys())
#             return classes, self.forced_class_to_idx
#         return super().find_classes(directory)

# --- CUSTOM DATASET CLASS (The Robust Fix) ---
class FixedLabelsImageFolder(datasets.ImageFolder):
    """
    Overrides ImageFolder to force a specific class-to-index mapping.
    CRITICAL FIX: It checks if the folder actually exists in the target directory
    before trying to load it. This prevents crashes when 'train' has a class 
    (like a garbage symbol '&') that 'val' does not.
    """
    def __init__(self, root, transform=None, class_to_idx=None):
        self.forced_class_to_idx = class_to_idx
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        if self.forced_class_to_idx is not None:
            # 1. Scan the directory to see which folders ACTUALLY exist here
            # (e.g., 'val' might have 'A', 'B' but miss '&')
            actual_folders = {d.name for d in os.scandir(directory) if d.is_dir()}
            
            # 2. Filter the Master Mapping
            # Keep the original index (e.g., A=0), but only if the folder exists here.
            filtered_class_to_idx = {
                cls_name: idx 
                for cls_name, idx in self.forced_class_to_idx.items() 
                if cls_name in actual_folders
            }
            
            classes = list(filtered_class_to_idx.keys())
            return classes, filtered_class_to_idx
            
        return super().find_classes(directory)

# --- 1. DATA TRANSFORMS ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(
            degrees=10, scale=(0.8, 1.2), translate=(0.1, 0.1), shear=5
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 2. SETUP DATASETS WITH FIXED LABELS ---

# A. Load Train first to establish the "Master" class list
train_dir = os.path.join(DATA_DIR, 'train')
full_train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])

# Capture the mapping! {'A': 0, 'B': 1, ...}
MASTER_CLASS_TO_IDX = full_train_dataset.class_to_idx
CLASS_NAMES = full_train_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)

print(f"Training on {DEVICE} with {NUM_CLASSES} classes.")
print(f"Classes found: {CLASS_NAMES}")

# B. Create the datasets
image_datasets = {
    'train': FixedLabelsImageFolder(train_dir, data_transforms['train'], class_to_idx=MASTER_CLASS_TO_IDX),
    'val': FixedLabelsImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val'], class_to_idx=MASTER_CLASS_TO_IDX)
}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# --- 3. MODEL SETUP ---
def initialize_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model.to(DEVICE)

# --- 4. TRAINING LOOP ---
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # WRAP WITH TQDM FOR PROGRESS BAR
            pbar = tqdm(dataloaders[phase], desc=phase, leave=False)

            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                pbar.set_postfix({'loss': loss.item()})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# --- 5. EXECUTION ---
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    model_ft = initialize_model()
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    model_ft, hist = train_model(model_ft, criterion, optimizer_ft, num_epochs=NUM_EPOCHS)
    
    torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # --- CONFUSION MATRIX ---
    print("Generating Confusion Matrix...")
    model_ft.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(DEVICE)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15,15))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(CM_SAVE_PATH)
    print(f"Confusion matrix saved to {CM_SAVE_PATH}")