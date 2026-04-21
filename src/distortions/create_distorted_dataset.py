import os
import random
import cv2
import pandas as pd
from tqdm import tqdm

from distortions.distortions import apply_distortion


def build_image_path(row, image_root: str) -> str:
    """
    Build image path from subject_id, study_id, dicom_id
    using MIMIC-style folder structure.
    """
    subject_id = str(int(row["subject_id"]))
    study_id = str(int(row["study_id"]))
    dicom_id = str(row["dicom_id"])
    prefix = subject_id[:2]

    return os.path.join(
        image_root,
        f"p{prefix}",
        f"p{subject_id}",
        f"s{study_id}",
        f"{dicom_id}.jpg",
    )


def build_save_path(row, output_root: str) -> str:
    subject_id = str(int(row["subject_id"]))
    study_id = str(int(row["study_id"]))
    dicom_id = str(row["dicom_id"])
    prefix = subject_id[:2]

    return os.path.join(
        output_root,
        f"p{prefix}",
        f"p{subject_id}",
        f"s{study_id}",
        f"{dicom_id}.jpg",
    )


def create_distorted_test_dataset(
    df: pd.DataFrame,
    image_root: str,
    output_root: str,
    distortions=("gaussian_noise", "motion_blur", "contrast_reduction"),
    severities=(1, 2),
):
    """
    Create separate distorted datasets for each distortion and severity.

    Output structure:
        output_root/
            gaussian_noise_s1/
            gaussian_noise_s2/
            motion_blur_s1/
            motion_blur_s2/
            contrast_reduction_s1/
            contrast_reduction_s2/
    """
    os.makedirs(output_root, exist_ok=True)

    total = len(df)
    print(f"Creating distorted datasets for {total} images")

    missing_count = 0
    unreadable_count = 0
    saved_count = 0

    for distortion_name in distortions:
        for severity in severities:
            dataset_root = os.path.join(output_root, f"{distortion_name}_s{severity}")
            os.makedirs(dataset_root, exist_ok=True)

            print(f"\nProcessing {distortion_name}_s{severity}")

            for _, row in tqdm(df.iterrows(), total=total):
                src_path = build_image_path(row, image_root)

                if not os.path.exists(src_path):
                    missing_count += 1
                    continue

                image = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    unreadable_count += 1
                    continue

                distorted = apply_distortion(image, distortion_name, severity)

                save_path = build_save_path(row, dataset_root)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                ok = cv2.imwrite(save_path, distorted)
                if ok:
                    saved_count += 1

    print("\nDone.")
    print(f"Saved images: {saved_count}")
    print(f"Missing source images skipped: {missing_count}")
    print(f"Unreadable source images skipped: {unreadable_count}")


def create_mixed_distortion_dataset(
    df: pd.DataFrame,
    image_root: str,
    output_root: str,
    distortions=("gaussian_noise", "motion_blur", "contrast_reduction"),
    severities=(1, 2),
    save_log: bool = True,
):
    """
    Create one mixed dataset where each image gets one randomly chosen
    distortion and severity.
    """
    os.makedirs(output_root, exist_ok=True)

    total = len(df)
    print(f"Creating mixed distortion dataset for {total} images")

    log = []
    missing_count = 0
    unreadable_count = 0
    saved_count = 0

    for _, row in tqdm(df.iterrows(), total=total):
        distortion_name = random.choice(distortions)
        severity = random.choice(severities)

        src_path = build_image_path(row, image_root)

        if not os.path.exists(src_path):
            missing_count += 1
            continue

        image = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            unreadable_count += 1
            continue

        distorted = apply_distortion(image, distortion_name, severity)

        save_path = build_save_path(row, output_root)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ok = cv2.imwrite(save_path, distorted)
        if ok:
            saved_count += 1
            log.append({
                "subject_id": int(row["subject_id"]),
                "study_id": int(row["study_id"]),
                "dicom_id": str(row["dicom_id"]),
                "distortion": distortion_name,
                "severity": severity,
            })

    if save_log:
        log_df = pd.DataFrame(log)
        log_path = os.path.join(output_root, "distortion_log.csv")
        log_df.to_csv(log_path, index=False)
        print(f"Log saved at: {log_path}")

    print("\nDone.")
    print(f"Saved images: {saved_count}")
    print(f"Missing source images skipped: {missing_count}")
    print(f"Unreadable source images skipped: {unreadable_count}")


if __name__ == "__main__":
    
    test_csv_path = "D:/HAMY_NET/hamy-net-xray-model-robustness/data/test_split.csv"
    image_root = "D:/HAMY_NET/hamy-net-xray-model-robustness/data/files"
    output_root = "D:/HAMY_NET/hamy-net-xray-model-robustness/data/distorted_test_files"
    mixed_output_root = "D:/HAMY_NET/hamy-net-xray-model-robustness/data/distorted_test_files/mixed_distortion"

    df = pd.read_csv(test_csv_path)

    # Separate distortion datasets
    create_distorted_test_dataset(
        df=df,
        image_root=image_root,
        output_root=output_root,
        distortions=("gaussian_noise", "motion_blur", "contrast_reduction"),
        severities=(1, 2),
    )

    # Mixed distortion dataset
    create_mixed_distortion_dataset(
        df=df,
        image_root=image_root,
        output_root=mixed_output_root,
        distortions=("gaussian_noise", "motion_blur", "contrast_reduction"),
        severities=(1, 2),
        save_log=True,
    )