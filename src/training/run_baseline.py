import torch
from torch.utils.data import DataLoader

from src.models.densenet121 import DenseNet121
from src.training.train import train_one_epoch
from src.evaluation.metrics import evaluate
from src.data.dataset import get_datasets  # assuming you have this

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset, val_dataset = get_datasets()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model
    model = DenseNet121(num_classes=1).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train
    epochs = 5
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")

    # Evaluate
    auroc = evaluate(model, val_loader, device)
    print(f"\n🔥 BASELINE AUROC: {auroc:.4f}")

    # Save model
    torch.save(model.state_dict(), "baseline_densenet121.pth")


if __name__ == "__main__":
    run()
