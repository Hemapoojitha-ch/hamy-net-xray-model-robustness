import pandas as pd
import numpy as np
from PIL import Image
import os

train_df = pd.read_csv("data/train_split.csv")

print("=" * 60)
print("DATA DIAGNOSTICS")
print("=" * 60)

# 1. Check class balance
print("\n1. Class Distribution:")
print(train_df['Pneumonia'].value_counts())
print(f"Positive ratio: {train_df['Pneumonia'].mean():.2%}")

# 2. Check for missing images
print("\n2. Image Availability:")
missing_count = 0
for idx, row in train_df.iterrows():
    subject_id = int(row['subject_id'])
    study_id = int(row['study_id'])
    dicom_id = row['dicom_id']
    prefix = str(subject_id)[:2]
    img_path = f"data/files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
    if not os.path.exists(img_path):
        missing_count += 1

print(f"Missing images: {missing_count}/{len(train_df)}")

# 3. Check image sizes
print("\n3. Image Dimensions:")
sizes = []
for idx in range(min(100, len(train_df))):  # Check first 100
    row = train_df.iloc[idx]
    subject_id = int(row['subject_id'])
    study_id = int(row['study_id'])
    dicom_id = row['dicom_id']
    prefix = str(subject_id)[:2]
    img_path = f"data/files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
    
    if os.path.exists(img_path):
        img = Image.open(img_path)
        sizes.append(img.size)

if sizes:
    print(f"Sample sizes: {set(sizes)}")
