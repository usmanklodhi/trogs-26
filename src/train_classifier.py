# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# import os
# import time
# import copy
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # --- CONFIGURATION ---
# DATA_DIR = "data/processed/greek_letters_classification"
# MODEL_SAVE_PATH = "models/greek_classifier_resnet18.pth"
# BATCH_SIZE = 32
# NUM_EPOCHS = 25
# LEARNING_RATE = 0.001
# NUM_CLASSES = len(os.listdir(os.path.join(DATA_DIR, 'train'))) # Auto-detect
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(f"Training on {DEVICE} with {NUM_CLASSES} classes.")

# # --- 1. DATA TRANSFORMS (The Robustness Layer) ---
# data_transforms = {
#     'train': transforms.Compose([
#         # Simulate YOLO Errors:
#         transforms.RandomAffine(
#             degrees=10,          # Rotation: +/- 10 degrees (simulates bad angle)
#             translate=(0.1, 0.1),# Shift: Move image 10% (simulates off-center)
#             scale=(0.8, 1.2),    # Zoom: +/- 20% (simulates box size error)
#             shear=5              # Shear: Perspective tilt
#         ),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2), # Stone lighting variance
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         # Normalize with ImageNet mean/std (standard for pre-trained models)
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# # --- 2. DATA LOADERS ---
# image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
#                   for x in ['train', 'val']}

# dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
#                for x in ['train', 'val']}

# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes

# # --- 3. MODEL SETUP ---
# def initialize_model():
#     # Load ResNet18 pre-trained on ImageNet
#     model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
#     # ResNet was trained on 1000 classes. We need to change the last layer.
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
#     return model.to(DEVICE)

# # --- 4. TRAINING LOOP ---
# def train_model(model, criterion, optimizer, num_epochs=25):
#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
    
#     history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(DEVICE)
#                 labels = labels.to(DEVICE)

#                 optimizer.zero_grad()

#                 # Forward pass
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # Backward pass + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
#             # Save history
#             if phase == 'train':
#                 history['train_loss'].append(epoch_loss)
#                 history['train_acc'].append(epoch_acc.item())
#             else:
#                 history['val_loss'].append(epoch_loss)
#                 history['val_acc'].append(epoch_acc.item())

#             # Deep copy the model if it's the best one so far
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#     time_elapsed = time.time() - since
#     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best val Acc: {best_acc:4f}')

#     # Load best model weights
#     model.load_state_dict(best_model_wts)
#     return model, history

# # --- 5. EXECUTION ---
# if __name__ == "__main__":
#     # Setup directories
#     os.makedirs("models", exist_ok=True)
    
#     # Init
#     model_ft = initialize_model()
#     criterion = nn.CrossEntropyLoss()
    
#     # Optimizer (SGD with momentum is standard, Adam is also fine)
#     optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
#     # Train
#     model_ft, hist = train_model(model_ft, criterion, optimizer_ft, num_epochs=NUM_EPOCHS)
    
#     # Save
#     torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)
#     print(f"Model saved to {MODEL_SAVE_PATH}")
    
#     # --- VISUALIZATION (Confusion Matrix) ---
#     # Good to see which letters confuse the model (e.g., A vs L)
#     print("Generating Confusion Matrix...")
#     model_ft.eval()
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for inputs, labels in dataloaders['val']:
#             inputs = inputs.to(DEVICE)
#             outputs = model_ft(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.numpy())
            
#     cm = confusion_matrix(all_labels, all_preds)
#     plt.figure(figsize=(15,15))
#     sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.savefig('models/confusion_matrix.png')
#     print("Confusion matrix saved.")