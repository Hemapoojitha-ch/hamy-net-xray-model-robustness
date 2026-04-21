import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import os
import json

from src.models.densenet121_baseline import DenseNet121Baseline
from src.evaluation.metrics import evaluate
from src.training.train import train_one_epoch
from src.data.dataset import ChestXrayDataset
from src.data.transforms_baseline import train_transform, val_transform

print("="*80)
print("BASELINE: DenseNet121 on Normalized Images (No Augmentation)")
print("="*80)

# ===== SIMPLE CONFIG =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-5

print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LR}")
print(f"  Weight Decay: {WEIGHT_DECAY}")

# ===== LOAD DATA =====
train_df = pd.read_csv("data/train_split.csv")
val_df = pd.read_csv("data/val_split.csv")
test_df = pd.read_csv("data/test_split.csv")

IMG_DIR = "data/images_normalized"

print(f"\nDataset:")
print(f"  Train: {len(train_df)} samples ({train_df['Pneumonia'].sum():.0f} positive)")
print(f"  Val:   {len(val_df)} samples ({val_df['Pneumonia'].sum():.0f} positive)")
print(f"  Test:  {len(test_df)} samples ({test_df['Pneumonia'].sum():.0f} positive)")

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
model = DenseNet121Baseline(weights='DEFAULT').to(DEVICE)

# Class weights
num_positive = train_df['Pneumonia'].sum()
num_negative = len(train_df) - num_positive
pos_weight = torch.tensor([num_negative / num_positive]).to(DEVICE)

print(f"\nClass Weight (Positive): {pos_weight.item():.3f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ===== CREATE OUTPUT DIR =====
os.makedirs("results/baseline", exist_ok=True)

# ===== TRAINING LOOP =====
best_auroc = 0.0
history = {'epoch': [], 'train_loss': [], 'val_auroc': [], 'val_acc': []}

print(f"\n{'='*80}")
print("TRAINING")
print(f"{'='*80}")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_auroc, val_acc, val_prec, val_rec, val_f1, val_spec, val_sens = evaluate(
        model, val_loader, DEVICE, verbose=False
    )

    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val AUROC: {val_auroc:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

    history['epoch'].append(epoch + 1)
    history['train_loss'].append(train_loss)
    history['val_auroc'].append(val_auroc)
    history['val_acc'].append(val_acc)

    scheduler.step()

    # Save best model
    if val_auroc > best_auroc:
        best_auroc = val_auroc
        torch.save(model.state_dict(), "results/baseline/densenet_baseline_best.pth")
        print(f"  ✅ New best AUROC: {best_auroc:.4f}")

# ===== FINAL EVALUATION =====
print(f"\n{'='*80}")
print("BASELINE EVALUATION ON TEST SET")
print(f"{'='*80}\n")

model.load_state_dict(torch.load("results/baseline/densenet_baseline_best.pth"))
test_auroc, test_acc, test_prec, test_rec, test_f1, test_spec, test_sens = evaluate(
    model, test_loader, DEVICE, verbose=True
)

# ===== RESULTS SUMMARY =====
print(f"\n{'='*80}")
print("BASELINE RESULTS")
print(f"{'='*80}")
print(f"\n📊 Best Validation AUROC: {best_auroc:.4f}")
print(f"\n📊 Test Set Metrics:")
print(f"     AUROC:       {test_auroc:.4f}")
print(f"     Accuracy:    {test_acc:.4f}")
print(f"     Precision:   {test_prec:.4f}")
print(f"     Recall:      {test_rec:.4f}")
print(f"     F1 Score:    {test_f1:.4f}")
print(f"     Sensitivity: {test_sens:.4f}")
print(f"     Specificity: {test_spec:.4f}")

# ===== SAVE RESULTS =====
results = {
    "model": "DenseNet121 Baseline",
    "config": {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "augmentation": "None (minimal transforms only)"
    },
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

with open("results/baseline/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n Baseline results saved to results/baseline/baseline_results.json")
print("="*80)
