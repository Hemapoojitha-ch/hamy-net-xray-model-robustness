import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
from pathlib import Path

# Load your CSV file
csv_path = 'data/mimic_ed_cxr_pneumonia_multimodal_cohort.csv'
df = pd.read_csv(csv_path)

# Set random seed for reproducibility
np.random.seed(42)

# Stratified split to maintain class balance (Pneumonia 32.79%, Non-Pneumonia 67.21%)
# First split: 70% train, 30% temp (which will be split into 15% test, 15% val)
train_df, temp_df = train_test_split(
    df, 
    test_size=0.30, 
    random_state=42,
    stratify=df['Pneumonia']  # Stratify by pneumonia label
)

# Second split: Split temp into test and val (50/50 of the 30%)
test_df, val_df = train_test_split(
    temp_df, 
    test_size=0.50, 
    random_state=42,
    stratify=temp_df['Pneumonia']
)

print("Dataset Split Summary:")
print(f"Total samples: {len(df)}")
print(f"Train set: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Test set:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
print(f"Val set:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print()
print("Class distribution:")
print(f"Train - Positive: {train_df['Pneumonia'].sum()}, Negative: {(train_df['Pneumonia']==0).sum()}")
print(f"Test  - Positive: {test_df['Pneumonia'].sum()}, Negative: {(test_df['Pneumonia']==0).sum()}")
print(f"Val   - Positive: {val_df['Pneumonia'].sum()}, Negative: {(val_df['Pneumonia']==0).sum()}")

# Save split CSVs
train_df.to_csv('data/train_split.csv', index=False)
test_df.to_csv('data/test_split.csv', index=False)
val_df.to_csv('data/val_split.csv', index=False)

print("\n✓ CSV splits saved:")
print("  - data/train_split.csv")
print("  - data/test_split.csv")
print("  - data/val_split.csv")

# Optional: Create directories and organize image files by split
def organize_images_by_split(df, split_name):
    """
    Create directories and copy/link images organized by split
    """
    split_dir = f'data/images_{split_name}'
    os.makedirs(split_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        
        # Get prefix from subject_id
        prefix = str(subject_id)[:2]
        
        src_path = f'data/files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg'
        
        if os.path.exists(src_path):
            # Create subdirectories
            dst_dir = f'{split_dir}/p{prefix}/p{subject_id}/s{study_id}'
            os.makedirs(dst_dir, exist_ok=True)
            
            dst_path = f'{dst_dir}/{dicom_id}.jpg'
            # You can use shutil.copy or os.symlink depending on your needs
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Image not found at {src_path}")

# Uncomment to organize images (may take time for large datasets)
# print("\nOrganizing images by split...")
# organize_images_by_split(train_df, 'train')
# organize_images_by_split(test_df, 'test')
# organize_images_by_split(val_df, 'val')
# print("✓ Images organized in data/images_train/, data/images_test/, data/images_val/")
