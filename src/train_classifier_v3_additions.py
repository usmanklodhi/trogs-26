import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
import copy
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from collections import Counter

# --- CONFIGURATION ---
DATA_DIR = "data/processed/greek_letters_classification_stratified"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = f"models/greek_classifier_resnet18_v4_{timestamp}.pth"
CM_SAVE_PATH = f"models/confusion_matrix_v4_{timestamp}.png"
REPORT_SAVE_PATH = f"models/classification_report_v4_{timestamp}.txt"

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM DATASET CLASS ---
class FixedLabelsImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, class_to_idx=None):
        self.forced_class_to_idx = class_to_idx
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        if self.forced_class_to_idx is not None:
            actual_folders = {d.name for d in os.scandir(directory) if d.is_dir()}
            filtered_class_to_idx = {
                cls_name: idx 
                for cls_name, idx in self.forced_class_to_idx.items() 
                if cls_name in actual_folders
            }
            classes = list(filtered_class_to_idx.keys())
            return classes, filtered_class_to_idx
        return super().find_classes(directory)

# --- 1. MODERATE TRANSFORMS (V4) ---
data_transforms = {
    'train': transforms.Compose([
        # Geometric - Softened to prevent distorting letters too much
        transforms.RandomAffine(
            degrees=10,          
            scale=(0.9, 1.1),    
            translate=(0.05, 0.05),
            shear=5             
        ),
        # Color/Texture augmentations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # RandomErasing REMOVED
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 2. SETUP DATASETS ---
train_dir = os.path.join(DATA_DIR, 'train')
full_train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
MASTER_CLASS_TO_IDX = full_train_dataset.class_to_idx
CLASS_NAMES = full_train_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)

print(f"Training on {DEVICE} with {NUM_CLASSES} classes.")

# Calculate Class Weights to handle Imbalance
print("Calculating softened class weights...")
train_targets = full_train_dataset.targets
class_counts = Counter(train_targets)
total_samples = len(train_targets)

# V4: Use Square Root to soften the extreme weights!
class_weights = []
for i in range(NUM_CLASSES):
    count = class_counts.get(i, 0)
    if count > 0:
        raw_weight = total_samples / (NUM_CLASSES * count)
        weight = np.sqrt(raw_weight)  # <--- THE MAGIC FIX
    else:
        weight = 1.0 
    class_weights.append(weight)

# Convert to tensor
class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
print(f"Class Weights applied (First 5): {class_weights[:5]}")

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
    
    # Add Dropout to prevent overfitting, but slightly reduced
    model.fc = nn.Sequential(
        nn.Dropout(0.4), 
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    return model.to(DEVICE)

# --- 4. TRAINING LOOP ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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

            if phase == 'train':
                scheduler.step() # Update learning rate

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Deep copy the model
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
    
    # USE WEIGHTED LOSS
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Add Weight Decay (L2 Regularization) to Optimizer
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    
    # Learning Rate Scheduler (StepLR)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    model_ft, hist = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS)
    
    torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # --- EVALUATION ---
    print("Generating Classification Report...")
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
            
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20,20))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(CM_SAVE_PATH)
    print(f"Confusion matrix saved to {CM_SAVE_PATH}")

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)
    print(report)
    with open(REPORT_SAVE_PATH, "w") as f:
        f.write(report)
    print(f"Report saved to {REPORT_SAVE_PATH}")