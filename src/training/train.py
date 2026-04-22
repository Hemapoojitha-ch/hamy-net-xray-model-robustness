import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
