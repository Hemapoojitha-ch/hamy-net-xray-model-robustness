import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
BATCH_SIZE = 16  # ↓ Reduced from 32 for better gradients
EPOCHS = 30  # ↑ Increased from 10
LR = 1e-3  # ↑ Increased learning rate
WEIGHT_DECAY = 1e-5  # L2 regularization
EARLY_STOPPING_PATIENCE = 5

print(f"Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"LR: {LR}")

# ===== LOAD DATA =====
train_df = pd.read_csv("data/train_split.csv")
val_df = pd.read_csv("data/val_split.csv")
test_df = pd.read_csv("data/test_split.csv")

IMG_DIR = "data/files"

print(f"\nDataset sizes:")
print(f"  Train: {len(train_df)} ({train_df['Pneumonia'].sum()} positive)")
print(f"  Val:   {len(val_df)} ({val_df['Pneumonia'].sum()} positive)")
print(f"  Test:  {len(test_df)} ({test_df['Pneumonia'].sum()} positive)")

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

# ↓ Use class weights if imbalanced
pos_weight = torch.tensor([len(train_df) / (2 * train_df['Pneumonia'].sum())])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True, min_lr=1e-6
)

# ===== CREATE OUTPUT DIR =====
os.makedirs("results/models", exist_ok=True)

# ===== TRAINING LOOP =====
best_auroc = 0.0
patience_counter = 0
history = {'epoch': [], 'train_loss': [], 'val_auroc': [], 'val_acc': []}

for epoch in range(EPOCHS):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*70}")

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_auroc, val_acc = evaluate(model, val_loader, DEVICE)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val AUROC: {val_auroc:.4f} | Val Acc: {val_acc:.4f}")

    # Store history
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(train_loss)
    history['val_auroc'].append(val_auroc)
    history['val_acc'].append(val_acc)

    # Learning rate scheduler step
    scheduler.step(val_auroc)

    # Early stopping & best model saving
    if val_auroc > best_auroc:
        best_auroc = val_auroc
        patience_counter = 0
        torch.save(model.state_dict(), "results/models/densenet_best.pth")
        print(f"✅ New best AUROC: {best_auroc:.4f} | Model saved")
    else:
        patience_counter += 1
        print(f"⚠️  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
        break

# ===== LOAD BEST MODEL & TEST =====
print(f"\n{'='*70}")
print("FINAL EVALUATION")
print(f"{'='*70}")

model.load_state_dict(torch.load("results/models/densenet_best.pth"))
test_auroc, test_acc = evaluate(model, test_loader, DEVICE)

print(f"\nTest Set Results:")
print(f"  AUROC: {test_auroc:.4f}")
print(f"  Accuracy: {test_acc:.4f}")

print(f"\nBest Validation AUROC: {best_auroc:.4f}")

# Save results
results = {
    "best_val_auroc": best_auroc,
    "test_auroc": test_auroc,
    "test_acc": test_acc,
    "history": history
}

with open("results/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n Results saved to results/training_results.json")
