import os
import torch
from torch.utils.data import DataLoader

from src.models.densenet121 import DenseNet121
from src.evaluation.metrics import evaluate
from src.data.distorted_dataset import DistortedDataset


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DenseNet121(num_classes=1).to(device)
    model.load_state_dict(torch.load("baseline_densenet121.pth"))
    model.eval()

    base_path = "data/distorted_test_files"
    csv_path = "data/test_split.csv"

    distortions = [
        "contrast_reduction_s2",
        "gaussian_noise_s2",
        "motion_blur_s2",
        "mixed_distortion",
    ]

    results = {}

    for distortion in distortions:
        print(f"\n🔹 Evaluating: {distortion}")

        dataset = DistortedDataset(
            csv_path=csv_path,
            distorted_root=os.path.join(base_path, distortion)
        )

        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        auroc = evaluate(model, loader, device)

        results[distortion] = auroc
        print(f"AUROC: {auroc:.4f}")

    print("\n📊 FINAL RESULTS:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    run()
