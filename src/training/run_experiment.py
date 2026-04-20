import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

# Local imports (IMPORTANT: use absolute imports from project root)
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



# DATASET 
import pandas as pd

train_df = pd.read_csv("data/train_split.csv")
val_df = pd.read_csv("data/val_split.csv")  

IMG_DIR = "data/images"


train_dataset = ChestXrayDataset(train_df, IMG_DIR, transform=train_transform)
val_dataset = ChestXrayDataset(val_df, IMG_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)



# MODEL
model = DenseNet121Binary(pretrained=True).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)



# TRAINING LOOP
best_auroc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

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
        print("✅ Saved best model")


print("\nTraining complete.")
print(f"Best AUROC: {best_auroc:.4f}")
