import os
import cv2
import matplotlib.pyplot as plt

from distortions import (
    gaussian_noise,
    motion_blur,
    contrast_reduction
)


image_paths = [
    "D:\HAMY_NET\hamy-net-xray-model-robustness\data\\files\p10\p10003019\s50543252\\3f4a324f-7967a6b4-91edf0c8-94fbefd4-32402065.jpg",
    "D:\HAMY_NET\hamy-net-xray-model-robustness\data\\files\p10\p10003956\s53245562\dc29d33e-bcf77ecf-c4fca6b6-8ea2ed29-d71aee14.jpg"
]

def show_results(img, title):
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")

for img_path in image_paths:
    print(f"\nTesting: {img_path}")

    if not os.path.exists(img_path):
        print("Image not found")
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Could not read image")
        continue

    # Apply distortions
    results = {
        "Original": img,
        "Noise s1": gaussian_noise(img, 1),
        "Noise s2": gaussian_noise(img, 2),
        "Blur s1": motion_blur(img, 1),
        "Blur s2": motion_blur(img, 2),
        "Contrast s1": contrast_reduction(img, 1),
        "Contrast s2": contrast_reduction(img, 2),
    }

    # Plot
    plt.figure(figsize=(12, 6))
    for i, (name, im) in enumerate(results.items()):
        plt.subplot(2, 4, i + 1)
        show_results(im, name)

    plt.suptitle(f"Distortion Test: {os.path.basename(img_path)}")
    plt.tight_layout()
    plt.show()