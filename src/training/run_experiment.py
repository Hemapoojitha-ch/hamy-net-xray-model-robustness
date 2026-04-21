import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import os
import json

from src.models.densenet121 import DenseNet121Binary
from src.evaluation.metrics import evaluate
from src.training.train import train_one_epoch
from src.data.dataset import ChestXrayDataset
from src.data.transforms import train_transform, val_transform

# ===== CONFIG =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 24
EPOCHS = 50
LR = 2e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 8

print(f"Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"LR: {LR}")

# ===== LOAD DATA =====
train_df = pd.read_csv("data/train_split.csv")
val_df = pd.read_csv("data/val_split.csv")
test_df = pd.read_csv("data/test_split.csv")

# Use normalized images
IMG_DIR = "data/images_normalized"

print(f"\nDataset sizes:")
print(f"  Train: {len(train_df)} ({train_df['Pneumonia'].sum():.0f} positive)")
print(f"  Val:   {len(val_df)} ({val_df['Pneumonia'].sum():.0f} positive)")
print(f"  Test:  {len(test_df)} ({test_df['Pneumonia'].sum():.0f} positive)")

# ===== CREATE DATASETS & LOADERS =====
train_dataset = ChestXrayDataset(train_df, IMG_DIR, transform=train_transform)
val_dataset = ChestXrayDataset(val_df, IMG_DIR, transform=val_transform)
test_dataset = ChestXrayDataset(test_df, IMG_DIR, transform=val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ===== MODEL & OPTIMIZER =====
model = DenseNet121Binary(weights='DEFAULT').to(DEVICE)

# Class weights for imbalanced data
num_positive = train_df['Pneumonia'].sum()
num_negative = len(train_df) - num_positive
pos_weight = torch.tensor([num_negative / num_positive]).to(DEVICE)

print(f"\nClass weights - Positive: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Cosine annealing scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ===== CREATE OUTPUT DIR =====
os.makedirs("results/models", exist_ok=True)

# ===== TRAINING LOOP =====
best_auroc = 0.0
patience_counter = 0
history = {'epoch': [], 'train_loss': [], 'val_auroc': [], 'val_acc': []}

for epoch in range(EPOCHS):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"{'='*70}")

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_auroc, val_acc = evaluate(model, val_loader, DEVICE)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val AUROC: {val_auroc:.4f} | Val Acc: {val_acc:.4f}")

    history['epoch'].append(epoch + 1)
    history['train_loss'].append(train_loss)
    history['val_auroc'].append(val_auroc)
    history['val_acc'].append(val_acc)

    scheduler.step()

    if val_auroc > best_auroc:
        best_auroc = val_auroc
        patience_counter = 0
        torch.save(model.state_dict(), "results/models/densenet_best.pth")
        print(f"New best AUROC: {best_auroc:.4f}")
    else:
        patience_counter += 1
        if patience_counter % 2 == 0:
            print(f" No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n Early stopping at epoch {epoch+1}")
        break

# ===== FINAL EVALUATION =====
print(f"\n{'='*70}")
print("FINAL EVALUATION")
print(f"{'='*70}")

model.load_state_dict(torch.load("results/models/densenet_best.pth"))
test_auroc, test_acc = evaluate(model, test_loader, DEVICE)

print(f"\n Results:")
print(f"  Best Val AUROC: {best_auroc:.4f}")
print(f"  Test AUROC:     {test_auroc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")

results = {
    "best_val_auroc": best_auroc,
    "test_auroc": test_auroc,
    "test_acc": test_acc,
    "history": history
}

with open("results/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n Done!")
