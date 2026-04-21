import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

from src.models.densenet121 import DenseNet121Binary
from src.evaluation.metrics import evaluate
from src.training.train import train_one_epoch
from src.data.dataset import ChestXrayDataset
from src.data.transforms import train_transform, val_transform

# ===== CONFIG =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16  # ↓ Smaller batch size
EPOCHS = 100
LR = 1e-4  # ↓ Lower learning rate
WEIGHT_DECAY = 1e-4  # ↑ Stronger L2 regularization
EARLY_STOPPING_PATIENCE = 15
DROPOUT_RATE = 0.5

print(f"Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"LR: {LR}")
print(f"Weight Decay (L2): {WEIGHT_DECAY}")
print(f"Dropout: {DROPOUT_RATE}")

# ===== LOAD DATA =====
train_df = pd.read_csv("data/train_split.csv")
val_df = pd.read_csv("data/val_split.csv")
test_df = pd.read_csv("data/test_split.csv")

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
model = DenseNet121Binary(weights='DEFAULT', dropout_rate=DROPOUT_RATE).to(DEVICE)

# Class weights
num_positive = train_df['Pneumonia'].sum()
num_negative = len(train_df) - num_positive
pos_weight = torch.tensor([num_negative / num_positive]).to(DEVICE)

print(f"\nClass weight - Positive: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ===== TWO-STAGE SCHEDULER =====
def get_scheduler(optimizer, epochs):
    """Warmup + Cosine Annealing"""
    warmup_epochs = 5
    main_epochs = epochs - warmup_epochs
    
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return (current_epoch + 1) / warmup_epochs
        else:
            progress = (current_epoch - warmup_epochs) / main_epochs
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

import numpy as np
scheduler = get_scheduler(optimizer, EPOCHS)

# ===== CREATE OUTPUT DIR =====
os.makedirs("results/models", exist_ok=True)

# ===== TRAINING LOOP =====
best_auroc = 0.0
patience_counter = 0
history = {
    'epoch': [], 'train_loss': [], 
    'val_auroc': [], 'val_acc': [], 'val_f1': [],
    'test_auroc': [], 'test_acc': []
}

for epoch in range(EPOCHS):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"{'='*80}")

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_auroc, val_acc, val_prec, val_rec, val_f1, val_spec, val_sens = evaluate(
        model, val_loader, DEVICE, verbose=True
    )

    print(f"\nTrain Loss: {train_loss:.4f}")

    history['epoch'].append(epoch + 1)
    history['train_loss'].append(train_loss)
    history['val_auroc'].append(val_auroc)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    scheduler.step()

    # ===== EARLY STOPPING =====
    if val_auroc > best_auroc:
        best_auroc = val_auroc
        patience_counter = 0
        torch.save(model.state_dict(), "results/models/densenet_best.pth")
        print(f" New best AUROC: {best_auroc:.4f}")
    else:
        patience_counter += 1
        if patience_counter % 3 == 0:
            print(f"  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n Early stopping at epoch {epoch+1}")
        break

# ===== FINAL EVALUATION =====
print(f"\n{'='*80}")
print("FINAL EVALUATION ON TEST SET")
print(f"{'='*80}\n")

model.load_state_dict(torch.load("results/models/densenet_best.pth"))
test_auroc, test_acc, test_prec, test_rec, test_f1, test_spec, test_sens = evaluate(
    model, test_loader, DEVICE, verbose=True
)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Best Val AUROC:  {best_auroc:.4f}")
print(f"Test AUROC:      {test_auroc:.4f}")
print(f"Test Accuracy:   {test_acc:.4f}")
print(f"Test F1 Score:   {test_f1:.4f}")
print(f"Test Sensitivity:{test_sens:.4f} (Recall/TPR)")
print(f"Test Specificity:{test_spec:.4f} (TNR)")

results = {
    "best_val_auroc": best_auroc,
    "test_metrics": {
        "auroc": test_auroc,
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "f1": test_f1,
        "sensitivity": test_sens,
        "specificity": test_spec
    },
    "history": history
}

with open("results/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ===== PLOT TRAINING HISTORY =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid()

axes[1].plot(history['epoch'], history['val_auroc'], 'r-', label='Val AUROC')
axes[1].plot(history['epoch'], history['val_f1'], 'g-', label='Val F1')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Score')
axes[1].set_title('Validation Metrics')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig("results/training_history.png", dpi=100)
print("\n Saved training plot to results/training_history.png")

print("\n Done!")
