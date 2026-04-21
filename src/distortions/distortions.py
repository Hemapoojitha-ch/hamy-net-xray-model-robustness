import cv2
import numpy as np


def gaussian_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Args:
        image: Input image as numpy array
        severity: 1 = mild, 2 = strong

    Returns:
        Distorted image
    """
    if severity not in [1, 2]:
        raise ValueError("severity must be 1 or 2")

    sigma_map = {
        1: 10.0,
        2: 20.0,
    }
    sigma = sigma_map[severity]

    image_f = image.astype(np.float32)
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = image_f + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _motion_blur_kernel(kernel_size: int, angle: float = 0.0) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0

    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))

    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def motion_blur(image: np.ndarray, severity: int = 1, angle: float = 0.0) -> np.ndarray:
    """
    Apply motion blur to an image.

    Args:
        image: Input image as numpy array
        severity: 1 = mild, 2 = strong
        angle: Blur direction angle in degrees

    Returns:
        Distorted image
    """
    if severity not in [1, 2]:
        raise ValueError("severity must be 1 or 2")

    kernel_size_map = {
        1: 7,
        2: 15,
    }
    kernel_size = kernel_size_map[severity]
    kernel = _motion_blur_kernel(kernel_size, angle=angle)

    blurred = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def contrast_reduction(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Reduce contrast by pulling pixel values toward the mean intensity.

    Args:
        image: Input image as numpy array
        severity: 1 = mild, 2 = strong

    Returns:
        Distorted image
    """
    if severity not in [1, 2]:
        raise ValueError("severity must be 1 or 2")

    factor_map = {
        1: 0.75,
        2: 0.50,
    }
    factor = factor_map[severity]

    image_f = image.astype(np.float32)

    if image.ndim == 2:
        mean = np.mean(image_f)
    else:
        mean = np.mean(image_f, axis=(0, 1), keepdims=True)

    reduced = mean + factor * (image_f - mean)
    return np.clip(reduced, 0, 255).astype(np.uint8)


def apply_distortion(image: np.ndarray, distortion_name: str, severity: int) -> np.ndarray:
    """
    Apply one supported distortion.

    Args:
        image: Input image
        distortion_name: gaussian_noise | motion_blur | contrast_reduction
        severity: 1 or 2

    Returns:
        Distorted image
    """
    if distortion_name == "gaussian_noise":
        return gaussian_noise(image, severity)
    elif distortion_name == "motion_blur":
        return motion_blur(image, severity)
    elif distortion_name == "contrast_reduction":
        return contrast_reduction(image, severity)
    else:
        raise ValueError(f"Unknown distortion: {distortion_name}")