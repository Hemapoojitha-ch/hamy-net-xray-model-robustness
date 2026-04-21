import torch
import pandas as pd
from src.models.densenet121_baseline import DenseNet121Baseline
from src.data.dataset import ChestXrayDataset
from src.data.transforms_baseline import val_transform
from src.evaluation.find_threshold import find_optimal_threshold
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load validation data
val_df = pd.read_csv("data/val_split.csv")
IMG_DIR = "data/images_normalized"

val_dataset = ChestXrayDataset(val_df, IMG_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load model
model = DenseNet121Baseline().to(DEVICE)
model.load_state_dict(torch.load("results/baseline/densenet_baseline_best.pth"))

# Find threshold
optimal_threshold, f1_scores, recalls, precisions = find_optimal_threshold(
    model, val_loader, DEVICE, metric='f1'
)

print(f"\n✅ Optimal Threshold (F1): {optimal_threshold:.3f}")
