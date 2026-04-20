import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df: DataFrame with 'dicom_id', 'Pneumonia' columns
            img_dir: Base directory containing images
            transform: Image transforms
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Construct image path (adjust based on your directory structure)
        dicom_id = row['dicom_id']
        subject_id = row['subject_id']
        study_id = row['study_id']
        
        img_path = f"{self.img_dir}/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['Pneumonia'], dtype=torch.float32)
        
        return image, label
