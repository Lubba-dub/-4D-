import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def cdf53_lifting_forward_1d(x):
    # Cohen-Daubechies-Feauveau 5/3 wavelet (Integer Lifting)
    # x: 1D array of integers
    x = x.astype(np.int32)
    n = len(x)
    
    # Pad if odd
    if n % 2 != 0:
        x = np.append(x, x[-1])
        n += 1
        
    s = x[0::2].copy() # Evens
    d = x[1::2].copy() # Odds
    
    # Predict (detail)
    # d[n] = d[n] - floor((s[n] + s[n+1])/2)
    # Handle boundary for s[n+1]: simple replication at boundary
    s_next = np.roll(s, -1)
    s_next[-1] = s[-1] # Boundary condition
    d -= np.floor((s + s_next) / 2).astype(np.int32)
    
    # Update (smooth)
    # s[n] = s[n] + floor((d[n-1] + d[n] + 2)/4)
    # Handle boundary for d[n-1]
    d_prev = np.roll(d, 1)
    d_prev[0] = d[0] # Boundary condition
    s += np.floor((d_prev + d + 2) / 4).astype(np.int32)
    
    return s, d

def cdf53_lifting_inverse_1d(s, d):
    s = s.astype(np.int32)
    d = d.astype(np.int32)
    
    # Inverse Update
    d_prev = np.roll(d, 1)
    d_prev[0] = d[0]
    s -= np.floor((d_prev + d + 2) / 4).astype(np.int32)
    
    # Inverse Predict
    s_next = np.roll(s, -1)
    s_next[-1] = s[-1]
    d += np.floor((s + s_next) / 2).astype(np.int32)
    
    # Interleave
    x_rec = np.zeros(2 * len(s), dtype=np.int32)
    x_rec[0::2] = s
    x_rec[1::2] = d
    
    return x_rec

def iwt2(img):
    # 2D Integer Wavelet Transform
    rows, cols = img.shape
    
    # Process Rows
    L = np.zeros((rows, cols//2), dtype=np.int32)
    H = np.zeros((rows, cols//2), dtype=np.int32)
    
    for i in range(rows):
        s, d = cdf53_lifting_forward_1d(img[i,:])
        L[i,:] = s
        H[i,:] = d
        
    # Process Cols on L and H
    LL = np.zeros((rows//2, cols//2), dtype=np.int32)
    LH = np.zeros((rows//2, cols//2), dtype=np.int32)
    HL = np.zeros((rows//2, cols//2), dtype=np.int32)
    HH = np.zeros((rows//2, cols//2), dtype=np.int32)
    
    for j in range(cols//2):
        s, d = cdf53_lifting_forward_1d(L[:,j])
        LL[:,j] = s
        LH[:,j] = d
        
        s, d = cdf53_lifting_forward_1d(H[:,j])
        HL[:,j] = s
        HH[:,j] = d
        
    return LL, LH, HL, HH

def iiwt2(LL, LH, HL, HH):
    # Inverse 2D IWT
    rows_half, cols_half = LL.shape
    rows = rows_half * 2
    cols = cols_half * 2
    
    L_rec = np.zeros((rows, cols_half), dtype=np.int32)
    H_rec = np.zeros((rows, cols_half), dtype=np.int32)
    
    # Inverse Cols
    for j in range(cols_half):
        L_rec[:,j] = cdf53_lifting_inverse_1d(LL[:,j], LH[:,j])
        H_rec[:,j] = cdf53_lifting_inverse_1d(HL[:,j], HH[:,j])
        
    img_rec = np.zeros((rows, cols), dtype=np.int32)
    
    # Inverse Rows
    for i in range(rows):
        img_rec[i,:] = cdf53_lifting_inverse_1d(L_rec[i,:], H_rec[i,:])
        
    return img_rec

def main():
    # Load Stage 1 result
    img_path = os.path.join("DIP_Project", "reports", "images", "stage1_roi_extracted.png")
    if not os.path.exists(img_path):
        print("Run Stage 1 first.")
        return
        
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure even dimensions for simplicity
    h, w = img.shape
    h = h - h%2
    w = w - w%2
    img = img[:h, :w]
    
    # 1. Forward IWT
    LL, LH, HL, HH = iwt2(img)
    
    # Visualization of decomposition
    coeffs_viz = np.vstack([
        np.hstack([LL, LH]),
        np.hstack([HL, HH])
    ])
    
    # 2. Inverse IWT
    img_rec = iiwt2(LL, LH, HL, HH)
    
    # 3. Verification
    diff = np.abs(img.astype(np.int32) - img_rec)
    max_diff = np.max(diff)
    print(f"Max reconstruction error: {max_diff}")
    
    if max_diff == 0:
        print("Verification Successful: Reconstruction is bit-perfect lossless.")
    else:
        print("Verification Failed: Lossy reconstruction.")

    # 4. Save & Plot
    save_dir = os.path.join("DIP_Project", "reports", "images")
    
    # Normalize coeffs for visualization
    coeffs_viz_norm = np.abs(coeffs_viz)
    coeffs_viz_norm = np.log1p(coeffs_viz_norm) # Log scale for better visibility
    coeffs_viz_norm = cv2.normalize(coeffs_viz_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    cv2.imwrite(os.path.join(save_dir, "stage2_iwt_decomposition.png"), coeffs_viz_norm)
    cv2.imwrite(os.path.join(save_dir, "stage2_reconstructed.png"), img_rec.astype(np.uint8))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original ROI')
    plt.subplot(1, 3, 2), plt.imshow(coeffs_viz_norm, cmap='gray'), plt.title('IWT Decomposition (CDF 5/3)')
    plt.subplot(1, 3, 3), plt.imshow(img_rec, cmap='gray'), plt.title(f'Reconstructed (Error={max_diff})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stage2_summary.png"))
    print("Stage 2 completed.")

if __name__ == "__main__":
    main()
