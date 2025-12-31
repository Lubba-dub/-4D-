import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def basic_otsu(image):
    # Otsu's thresholding
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret, thresh

def morphological_optimization(mask):
    # Kernel for morphological operations
    kernel = np.ones((5,5), np.uint8)
    
    # Closing to fill holes inside the object
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Opening to remove noise in the background
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    return opening

def extract_roi(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def main():
    # Load image (Use noisy phantom for better demonstration of morphology)
    img_path = os.path.join("DIP_Project", "data", "noisy_phantom.png")
    if not os.path.exists(img_path):
        img_path = os.path.join("DIP_Project", "data", "phantom.png")
        
    original = load_image(img_path)
    if original is None:
        print(f"Error loading image from {img_path}")
        return

    # 1. Basic Otsu
    otsu_thresh_val, otsu_mask = basic_otsu(original)
    
    # 2. Morphological Optimization
    refined_mask = morphological_optimization(otsu_mask)
    
    # 3. Extract ROI
    roi_image = extract_roi(original, refined_mask)
    
    # Save results
    save_dir = os.path.join("DIP_Project", "reports", "images")
    cv2.imwrite(os.path.join(save_dir, "stage1_original.png"), original)
    cv2.imwrite(os.path.join(save_dir, "stage1_otsu_mask.png"), otsu_mask)
    cv2.imwrite(os.path.join(save_dir, "stage1_refined_mask.png"), refined_mask)
    cv2.imwrite(os.path.join(save_dir, "stage1_roi_extracted.png"), roi_image)
    
    # Visualization for Report
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1), plt.imshow(original, cmap='gray'), plt.title('Original')
    plt.subplot(1, 4, 2), plt.imshow(otsu_mask, cmap='gray'), plt.title(f'Otsu (T={otsu_thresh_val:.1f})')
    plt.subplot(1, 4, 3), plt.imshow(refined_mask, cmap='gray'), plt.title('Morphological Refined')
    plt.subplot(1, 4, 4), plt.imshow(roi_image, cmap='gray'), plt.title('Extracted ROI')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stage1_summary.png"))
    print("Stage 1 completed. Images saved.")

if __name__ == "__main__":
    main()
