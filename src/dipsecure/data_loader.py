import os
from typing import List, Tuple
import numpy as np
import cv2

try:
    import medmnist
    from medmnist import INFO
except Exception:
    INFO = None
    medmnist = None


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_medmnist(dataset_name: str = "chestmnist", split: str = "test", max_items: int = 1000) -> List[np.ndarray]:
    """
    Load MedMNIST dataset images as grayscale numpy arrays.
    dataset_name: one of INFO.keys(), e.g., 'chestmnist', 'organmnist', 'pathmnist'
    split: 'train' | 'val' | 'test'
    max_items: limit number of images to process for speed
    """
    if INFO is None or medmnist is None:
        raise RuntimeError("medmnist is not installed correctly.")
    if dataset_name not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset: {dataset_name}")
    info = INFO[dataset_name]
    python_class = info["python_class"]
    DataClass = getattr(medmnist, python_class)
    dataset = DataClass(split=split, download=True)
    imgs = dataset.imgs  # numpy array HxWxC
    # Convert to grayscale uint8
    images: List[np.ndarray] = []
    for i in range(min(len(imgs), max_items)):
        img = imgs[i]
        if img.ndim == 3:
            # If RGB, convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        # Ensure uint8
        if gray.dtype != np.uint8:
            gray = (255 * np.clip(gray, 0, 1)).astype(np.uint8)
        images.append(gray)
    return images


def save_image(img: np.ndarray, path: str):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

