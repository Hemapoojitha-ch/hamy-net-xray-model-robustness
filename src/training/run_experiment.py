import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os

# Local imports
from src.models.densenet121 import DenseNet121Binary
from src.evaluation.metrics import evaluate
from src.training.train import train_one_epoch
from src.data.dataset import ChestXrayDataset
from src.data.transforms import train_transform, val_transform

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

print(f"Using device: {DEVICE}")

# DATASET
train_df = pd.read_csv("data/train_split.csv")
val_df = pd.read_csv("data/val_split.csv")
test_df = pd.read_csv("data/test_split.csv")

IMG_DIR = "data/files"  

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

train_dataset = ChestXrayDataset(train_df, IMG_DIR, transform=train_transform)
val_dataset = ChestXrayDataset(val_df, IMG_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# MODEL
model = DenseNet121Binary(weights='DEFAULT').to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Create results directory
os.makedirs("results/models", exist_ok=True)

# TRAINING LOOP
best_auroc = 0.0

for epoch in range(EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*60}")

    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE
    )

    val_auroc, val_acc = evaluate(
        model, val_loader, DEVICE
    )

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val AUROC: {val_auroc:.4f} | Val Acc: {val_acc:.4f}")

    # Save best model
    if val_auroc > best_auroc:
        best_auroc = val_auroc
        torch.save(model.state_dict(), "results/models/densenet_best.pth")
        print("Saved best model")

print("\n" + "="*60)
print("Training complete.")
print(f"Best AUROC: {best_auroc:.4f}")
print("="*60)
