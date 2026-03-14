# import os
# import time
# import copy
# import random
# from datetime import datetime

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader

# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.metrics import confusion_matrix, classification_report, f1_score


# # =========================================================
# # --- CONFIGURATION ---
# # =========================================================
# DATA_DIR = "data/processed/greek_letters_classification_stratified"

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# MODEL_SAVE_PATH = f"models/greek_classifier_resnet18_{timestamp}.pth"
# CM_SAVE_PATH = f"models/confusion_matrix_{timestamp}.png"
# REPORT_SAVE_PATH = f"models/classification_report_{timestamp}.txt"

# BATCH_SIZE = 32
# NUM_EPOCHS = 25

# # AdamW usually likes a smaller LR than SGD
# LEARNING_RATE = 3e-4
# WEIGHT_DECAY = 1e-4

# NUM_WORKERS = 4
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Reproducibility
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.benchmark = True  # speed (good for fixed-size 128x128)

# print(f"Training on: {DEVICE}")
# print(f"Model will be saved to: {MODEL_SAVE_PATH}")


# # =========================================================
# # --- CUSTOM DATASET CLASS (Robust Fix for missing folders) ---
# # =========================================================
# class FixedLabelsImageFolder(datasets.ImageFolder):
#     """
#     Overrides ImageFolder to force a specific class-to-index mapping.
#     Robust fix: checks which folders exist in the target directory and
#     filters the master mapping accordingly, keeping original indices.
#     """
#     def __init__(self, root, transform=None, class_to_idx=None):
#         self.forced_class_to_idx = class_to_idx
#         super().__init__(root, transform=transform)

#     def find_classes(self, directory):
#         if self.forced_class_to_idx is not None:
#             actual_folders = {d.name for d in os.scandir(directory) if d.is_dir()}

#             filtered_class_to_idx = {
#                 cls_name: idx
#                 for cls_name, idx in self.forced_class_to_idx.items()
#                 if cls_name in actual_folders
#             }

#             # Keep classes ordered by their MASTER index (important for consistency)
#             classes = [k for k, v in sorted(filtered_class_to_idx.items(), key=lambda kv: kv[1])]
#             return classes, filtered_class_to_idx

#         return super().find_classes(directory)


# # =========================================================
# # --- 1) DATA TRANSFORMS (adds RandomPerspective + RandomErasing) ---
# # =========================================================
# data_transforms = {
#     "train": transforms.Compose([
#         transforms.RandomAffine(
#             degrees=10, scale=(0.8, 1.2), translate=(0.1, 0.1), shear=5
#         ),
#         transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2),
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]),
#     "val": transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]),
# }


# # =========================================================
# # --- 2) SETUP DATASETS WITH FIXED LABELS ---
# # =========================================================
# train_dir = os.path.join(DATA_DIR, "train")
# val_dir = os.path.join(DATA_DIR, "val")

# # Load train first to create MASTER mapping
# full_train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
# MASTER_CLASS_TO_IDX = full_train_dataset.class_to_idx
# CLASS_NAMES = full_train_dataset.classes
# NUM_CLASSES = len(CLASS_NAMES)

# print(f"NUM_CLASSES (from train): {NUM_CLASSES}")
# print(f"Classes found: {CLASS_NAMES}")

# image_datasets = {
#     "train": FixedLabelsImageFolder(train_dir, transform=data_transforms["train"], class_to_idx=MASTER_CLASS_TO_IDX),
#     "val": FixedLabelsImageFolder(val_dir, transform=data_transforms["val"], class_to_idx=MASTER_CLASS_TO_IDX),
# }

# # NOTE: shuffle=False for val (important)
# dataloaders = {
#     "train": DataLoader(
#         image_datasets["train"],
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=torch.cuda.is_available(),
#         persistent_workers=(NUM_WORKERS > 0),
#     ),
#     "val": DataLoader(
#         image_datasets["val"],
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=torch.cuda.is_available(),
#         persistent_workers=(NUM_WORKERS > 0),
#     ),
# }

# dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}


# # =========================================================
# # --- 3) MODEL SETUP ---
# # =========================================================
# def initialize_model():
#     model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
#     return model.to(DEVICE)


# # =========================================================
# # --- 4) TRAINING LOOP (adds AMP + AdamW + Cosine scheduler) ---
# # =========================================================
# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

#     use_amp = (DEVICE.type == "cuda")
#     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
#         print("-" * 10)

#         for phase in ["train", "val"]:
#             if phase == "train":
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             pbar = tqdm(dataloaders[phase], desc=phase, leave=False)

#             for inputs, labels in pbar:
#                 inputs = inputs.to(DEVICE, non_blocking=True)
#                 labels = labels.to(DEVICE, non_blocking=True)

#                 optimizer.zero_grad(set_to_none=True)

#                 with torch.set_grad_enabled(phase == "train"):
#                     # AMP autocast on CUDA
#                     with torch.cuda.amp.autocast(enabled=use_amp):
#                         outputs = model(inputs)
#                         _, preds = torch.max(outputs, 1)
#                         loss = criterion(outputs, labels)

#                     if phase == "train":
#                         scaler.scale(loss).backward()
#                         scaler.step(optimizer)
#                         scaler.update()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data).item()

#                 pbar.set_postfix({"loss": f"{loss.item():.4f}"})

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects / dataset_sizes[phase]

#             print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

#             if phase == "train":
#                 history["train_loss"].append(epoch_loss)
#                 history["train_acc"].append(epoch_acc)
#                 # Step scheduler once per epoch (train only)
#                 scheduler.step()
#             else:
#                 history["val_loss"].append(epoch_loss)
#                 history["val_acc"].append(epoch_acc)

#                 if epoch_acc > best_acc:
#                     best_acc = epoch_acc
#                     best_model_wts = copy.deepcopy(model.state_dict())

#     time_elapsed = time.time() - since
#     print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
#     print(f"Best val Acc: {best_acc:.4f}")

#     model.load_state_dict(best_model_wts)
#     return model, history


# # =========================================================
# # --- 5) EXECUTION + CONFUSION MATRIX + CLASSIFICATION REPORT + MACRO F1 ---
# # =========================================================
# if __name__ == "__main__":
#     os.makedirs("models", exist_ok=True)

#     model_ft = initialize_model()
#     criterion = nn.CrossEntropyLoss()

#     # 1) AdamW optimizer
#     optimizer_ft = optim.AdamW(
#         model_ft.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY
#     )

#     # 2) Cosine LR scheduler
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=NUM_EPOCHS)

#     model_ft, hist = train_model(
#         model_ft,
#         criterion,
#         optimizer_ft,
#         scheduler,
#         num_epochs=NUM_EPOCHS
#     )

#     torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)
#     print(f"Model saved to {MODEL_SAVE_PATH}")

#     # --- EVALUATION: Confusion Matrix + Macro F1 + Classification Report ---
#     print("Generating Confusion Matrix + Classification Report...")
#     model_ft.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for inputs, labels in dataloaders["val"]:
#             inputs = inputs.to(DEVICE, non_blocking=True)
#             outputs = model_ft(inputs)
#             _, preds = torch.max(outputs, 1)

#             all_preds.extend(preds.cpu().numpy().tolist())
#             all_labels.extend(labels.numpy().tolist())

#     # Confusion Matrix
#     cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
#     plt.figure(figsize=(15, 15))
#     sns.heatmap(cm, annot=False, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     plt.savefig(CM_SAVE_PATH, dpi=200)
#     plt.close()
#     print(f"Confusion matrix saved to {CM_SAVE_PATH}")

#     # Macro F1
#     macro_f1 = f1_score(all_labels, all_preds, average="macro")
#     print(f"Macro F1 (val): {macro_f1:.4f}")

#     # Classification report (handles missing classes in val)
#     report = classification_report(
#         all_labels,
#         all_preds,
#         labels=list(range(NUM_CLASSES)),
#         target_names=CLASS_NAMES,
#         zero_division=0
#     )

#     with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
#         f.write(f"Timestamp: {timestamp}\n")
#         f.write(f"Val Accuracy (best checkpoint): {max(hist['val_acc']) if hist['val_acc'] else 'N/A'}\n")
#         f.write(f"Macro F1 (val): {macro_f1:.4f}\n\n")
#         f.write(report)

#     print(f"Classification report saved to {REPORT_SAVE_PATH}")