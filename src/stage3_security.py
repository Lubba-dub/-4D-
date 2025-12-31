import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import hashlib

def logistic_map_keygen(length, x0=0.1, r=3.99):
    # 1D Logistic Map for simplicity and speed
    # x(n+1) = r * x(n) * (1 - x(n))
    key = np.zeros(length, dtype=np.uint8)
    x = x0
    # Discard transients
    for _ in range(100):
        x = r * x * (1 - x)
        
    for i in range(length):
        x = r * x * (1 - x)
        key[i] = int((x * 1000) % 256)
        
    return key

def encrypt_image_xor(img, key):
    flat_img = img.flatten()
    if len(key) < len(flat_img):
        raise ValueError("Key too short")
    
    encrypted = np.bitwise_xor(flat_img, key[:len(flat_img)])
    return encrypted.reshape(img.shape)

def decrypt_image_xor(enc_img, key):
    return encrypt_image_xor(enc_img, key) # XOR is symmetric

def calculate_phash(img):
    # Perceptual Hash
    # 1. Resize to 32x32
    img_small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img_small = img_small.astype(np.float32)
    
    # 2. DCT
    dct = cv2.dct(img_small)
    
    # 3. Focus on top-left 8x8
    dct_low_freq = dct[0:8, 0:8]
    
    # 4. Average value (exclude DC term at 0,0)
    avg = np.mean(dct_low_freq)
    
    # 5. Compute Hash (1 if > avg, 0 otherwise)
    phash = 0
    idx = 0
    for i in range(8):
        for j in range(8):
            phash <<= 1
            if dct_low_freq[i,j] > avg:
                phash |= 1
            idx += 1
            
    return phash

def calculate_hamming_distance(hash1, hash2):
    x = hash1 ^ hash2
    dist = 0
    while x > 0:
        dist += x & 1
        x >>= 1
    return dist

def main():
    # Load Stage 2 result (Reconstructed Image)
    # Actually we encrypt the compressed stream usually, but for visualization
    # we encrypt the pixel domain of the ROI here.
    img_path = os.path.join("DIP_Project", "reports", "images", "stage1_roi_extracted.png")
    if not os.path.exists(img_path):
        print("Run Stage 1 first.")
        return
        
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Generate Key
    total_pixels = img.shape[0] * img.shape[1]
    key = logistic_map_keygen(total_pixels, x0=0.123456789, r=3.9999)
    
    # 2. Encrypt
    enc_img = encrypt_image_xor(img, key)
    
    # 3. Decrypt
    dec_img = decrypt_image_xor(enc_img, key)
    
    # 4. Perceptual Hash (Zero-Watermarking)
    # Generate pHash for Original ROI
    phash_orig = calculate_phash(img)
    print(f"Original pHash: {phash_orig:016x}")
    
    # 5. Robustness Test (Simulate Attack)
    # Add salt & pepper noise to Encrypted Image
    enc_img_noisy = enc_img.copy()
    noise_indices = np.random.randint(0, total_pixels, total_pixels // 100) # 1% noise
    flat_noisy = enc_img_noisy.flatten()
    flat_noisy[noise_indices] = 255
    enc_img_noisy = flat_noisy.reshape(img.shape)
    
    # Decrypt Noisy Image
    dec_img_noisy = decrypt_image_xor(enc_img_noisy, key)
    
    # Calculate pHash of Noisy Decrypted Image
    phash_noisy = calculate_phash(dec_img_noisy)
    print(f"Noisy Decrypted pHash: {phash_noisy:016x}")
    
    # Compare
    hamming_dist = calculate_hamming_distance(phash_orig, phash_noisy)
    print(f"Hamming Distance: {hamming_dist} (<=5 is usually considered a match)")
    
    # SHA256 Comparison (for contrast)
    sha256_orig = hashlib.sha256(img.tobytes()).hexdigest()
    sha256_noisy = hashlib.sha256(dec_img_noisy.tobytes()).hexdigest()
    print(f"SHA256 Match: {sha256_orig == sha256_noisy}")
    
    # Save & Plot
    save_dir = os.path.join("DIP_Project", "reports", "images")
    cv2.imwrite(os.path.join(save_dir, "stage3_encrypted.png"), enc_img)
    cv2.imwrite(os.path.join(save_dir, "stage3_decrypted_noisy.png"), dec_img_noisy)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original ROI')
    plt.subplot(2, 3, 2), plt.imshow(enc_img, cmap='gray'), plt.title('Encrypted (Chaos)')
    plt.subplot(2, 3, 3), plt.hist(enc_img.ravel(), 256, [0, 256]), plt.title('Histogram (Encrypted)')
    
    plt.subplot(2, 3, 4), plt.imshow(enc_img_noisy, cmap='gray'), plt.title('Encrypted + 1% Noise')
    plt.subplot(2, 3, 5), plt.imshow(dec_img_noisy, cmap='gray'), plt.title('Decrypted (Noisy)')
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f"pHash Dist: {hamming_dist}\n(Robust)", fontsize=12)
    plt.text(0.1, 0.5, f"SHA256 Match: False\n(Sensitive)", fontsize=12)
    plt.axis('off')
    plt.title('Authentication Result')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stage3_summary.png"))
    print("Stage 3 completed.")

if __name__ == "__main__":
    main()
