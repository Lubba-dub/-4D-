import numpy as np
import cv2
from skimage.filters import threshold_sauvola
from skimage.morphology import remove_small_objects, remove_small_holes, convex_hull_image


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def refine_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening


def extract_roi(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image, image, mask=mask)


def split_roi_background(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    roi = extract_roi(image, mask)
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    return roi, background


def robust_roi_mask(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(image)
    img_blur = cv2.medianBlur(img_eq, 5)
    sau = threshold_sauvola(img_blur, window_size=33, k=0.1)
    lung_bin = img_blur < sau
    lung_bin = remove_small_objects(lung_bin, min_size=int(0.003 * h * w))
    lung_bin = remove_small_holes(lung_bin, area_threshold=int(0.003 * h * w))
    mask_u8 = (lung_bin.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    areas = []
    for lbl in range(1, num_labels):
        x, y, bw, bh, area = stats[lbl, cv2.CC_STAT_LEFT], stats[lbl, cv2.CC_STAT_TOP], stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT], stats[lbl, cv2.CC_STAT_AREA]
        if x <= 0 or y <= 0 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        areas.append((area, lbl))
    if not areas:
        lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        sel = labels == lbl
    else:
        areas.sort(reverse=True)
        sel = (labels == areas[0][1])
        if len(areas) > 1:
            sel = np.logical_or(sel, labels == areas[1][1])
    hull = convex_hull_image(sel.astype(bool))
    return (hull.astype(np.uint8)) * 255

