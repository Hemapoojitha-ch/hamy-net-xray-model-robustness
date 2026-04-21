import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Load all splits
train_df = pd.read_csv("data/train_split.csv")
val_df = pd.read_csv("data/val_split.csv")
test_df = pd.read_csv("data/test_split.csv")

all_df = pd.concat([train_df, val_df, test_df])

# Target size
TARGET_SIZE = 256

# Create output directory
output_dir = "data/images_normalized"
os.makedirs(output_dir, exist_ok=True)

print(f"Processing {len(all_df)} images to {TARGET_SIZE}x{TARGET_SIZE}...")

for idx, row in tqdm(all_df.iterrows()):
    subject_id = int(row['subject_id'])
    study_id = int(row['study_id'])
    dicom_id = row['dicom_id']
    prefix = str(subject_id)[:2]
    
    # Source path
    src_path = f"data/files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
    
    # Create output directory structure
    out_subdir = os.path.join(output_dir, f"p{prefix}", f"p{subject_id}", f"s{study_id}")
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, f"{dicom_id}.jpg")
    
    if os.path.exists(src_path):
        # Read image
        img = cv2.imread(src_path)
        
        if img is not None:
            # Resize while preserving aspect ratio
            h, w = img.shape[:2]
            
            # Scale to fit in TARGET_SIZE x TARGET_SIZE while maintaining aspect ratio
            scale = TARGET_SIZE / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad to TARGET_SIZE x TARGET_SIZE
            pad_h = TARGET_SIZE - new_h
            pad_w = TARGET_SIZE - new_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            img_padded = cv2.copyMakeBorder(
                img_resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(128, 128, 128)
            )
            
            # Save
            cv2.imwrite(out_path, img_padded)

print(f"\n All images normalized and saved to {output_dir}")
