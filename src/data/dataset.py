import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df: DataFrame with image metadata
            img_dir: Base directory ('data/images_normalized' for preprocessed)
            transform: Image transforms
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        subject_id = int(row['subject_id'])
        study_id = int(row['study_id'])
        dicom_id = row['dicom_id']
        prefix = str(subject_id)[:2]
        
        img_path = os.path.join(
            self.img_dir, 
            f"p{prefix}", 
            f"p{subject_id}", 
            f"s{study_id}", 
            f"{dicom_id}.jpg"
        )
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"  Image not found: {img_path}")
            image = Image.new('RGB', (256, 256), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(float(row['Pneumonia']), dtype=torch.float32)
        
        return image, label
