import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DistortedDataset(Dataset):
    def __init__(self, csv_path, distorted_root):
        self.df = pd.read_csv(csv_path)
        self.root = distorted_root

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Adjust column names based on your CSV
        img_path = row["path"]     # or "image_path"
        label = row["label"]       # or "target"

        # Replace original path with distorted path
        distorted_path = os.path.join(self.root, img_path)

        image = Image.open(distorted_path).convert("RGB")
        image = self.transform(image)

        return image, label
