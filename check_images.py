import pandas as pd
import os

train_df = pd.read_csv("data/train_split.csv")

# Check first 5 images
for idx in range(min(5, len(train_df))):
    row = train_df.iloc[idx]
    subject_id = int(row['subject_id'])
    study_id = int(row['study_id'])
    dicom_id = row['dicom_id']
    prefix = str(subject_id)[:2]
    
    img_path = f"data/files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
    exists = os.path.exists(img_path)
    print(f"{'✅' if exists else '❌'} {img_path}")
