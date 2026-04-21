import torch
import pandas as pd
from src.models.densenet121_baseline import DenseNet121Baseline
from src.data.dataset import ChestXrayDataset
from src.data.transforms_baseline import val_transform
from src.evaluation.metrics import evaluate
from src.evaluation.find_threshold import find_optimal_threshold
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("TEST SET EVALUATION WITH DIFFERENT THRESHOLDS")
print("="*80)

# Load test data
test_df = pd.read_csv("data/test_split.csv")
IMG_DIR = "data/images_normalized"

test_dataset = ChestXrayDataset(test_df, IMG_DIR, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load model
model = DenseNet121Baseline().to(DEVICE)
model.load_state_dict(torch.load("results/baseline/densenet_baseline_best.pth"))

# Find optimal threshold on validation
val_df = pd.read_csv("data/val_split.csv")
val_dataset = ChestXrayDataset(val_df, IMG_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

optimal_threshold, _, _, _ = find_optimal_threshold(model, val_loader, DEVICE, metric='f1')

print(f"\n{'='*80}")
print(f"📊 At Threshold 0.5 (Baseline):")
print(f"{'='*80}")
evaluate(model, test_loader, DEVICE, threshold=0.5, verbose=True)

print(f"\n{'='*80}")
print(f"📊 At Optimal Threshold {optimal_threshold:.3f}:")
print(f"{'='*80}")
evaluate(model, test_loader, DEVICE, threshold=optimal_threshold, verbose=True)

print(f"\n{'='*80}")
print(f"📊 At Medical Threshold 0.3 (Prioritize Recall):")
print(f"{'='*80}")
evaluate(model, test_loader, DEVICE, threshold=0.3, verbose=True)

print(f"\n✅ Complete! Check results/baseline/threshold_analysis.png for visualization")
