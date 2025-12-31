import os
import hashlib
import cv2
import numpy as np
from typing import Dict
from dipsecure.data_loader import get_medmnist, save_image, ensure_dir
from dipsecure.roi import otsu_threshold, refine_mask, split_roi_background
from dipsecure.metrics import psnr, ssim, compression_bytes, entropy_bits, adaptive_bg_quality, npcr_uaci, zlib_compress_size, rle_zero_zlib_size, ll_diff_zlib_size
from dipsecure.iwt import iwt2, iiwt2
from dipsecure.chaos_crypto import logistic_key, xor_stream, logistic_sine_key, xor_diffuse_encrypt, xor_diffuse_decrypt, perm_diffuse_encrypt, perm_diffuse_decrypt, perm_bidiffuse_encrypt, perm_bidiffuse_decrypt
from dipsecure.zero_watermark import dwt_ll_signature, ecc_encode_repetition, ecc_decode_repetition, rs_encode, rs_decode
from dipsecure.phash import phash64, hamming
from dipsecure.blockchain import ImageChain, ChainConfig


REPORT_DIR = os.path.join("DIP_Project", "reports")
IMG_DIR = os.path.join(REPORT_DIR, "images")
ensure_dir(IMG_DIR)


def process_image(img: np.ndarray) -> Dict:
    mask = otsu_threshold(img)
    mask_ref = refine_mask(mask, kernel_size=5)
    roi, bg = split_roi_background(img, mask_ref)
    q_bg = adaptive_bg_quality(bg)
    comp = compression_bytes(roi, bg, mask_ref, q_bg=q_bg)

    # IWT demonstration + perfect reconstruction (on ROI)
    h, w = roi.shape
    roi_cropped = roi[: h - h % 2, : w - w % 2]
    LL, LH, HL, HH = iwt2(roi_cropped)
    roi_rec = iiwt2(LL, LH, HL, HH).astype(np.uint8)
    psnr_roi = psnr(roi_cropped, roi_rec)
    ssim_roi = ssim(roi_cropped, roi_rec)

    # Chaos encryption on whole image
    total_pixels = img.size
    key1 = logistic_sine_key(total_pixels)
    key2 = logistic_sine_key(total_pixels, x0=0.654321, y0=0.24680)
    enc = perm_bidiffuse_encrypt(img, key1, key2)

    # Robustness: JPEG re-encode PLAINTEXT (simulates transmission/format conversion)
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    img_jpeg = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    enc_noisy = perm_bidiffuse_encrypt(img_jpeg, key1, key2)
    dec_noisy = perm_bidiffuse_decrypt(enc_noisy, key1, key2)

    # Authentication
    ph_orig = phash64(img)
    ph_noisy = phash64(dec_noisy)
    dist = hamming(ph_orig, ph_noisy)
    sha_orig = hashlib.sha256(img.tobytes()).hexdigest()
    sha_noisy = hashlib.sha256(dec_noisy.tobytes()).hexdigest()

    # NPCR/UACI on single-pixel change
    img2 = img.copy()
    img2[img2.shape[0] // 2, img2.shape[1] // 2] = (img2[img2.shape[0] // 2, img2.shape[1] // 2] + 1) % 255
    enc_a = perm_bidiffuse_encrypt(img, key1, key2)
    enc_b = perm_bidiffuse_encrypt(img2, key1, key2)
    npcr, uaci = npcr_uaci(enc_a, enc_b)

    # Zero-watermark signature robustness
    zw_orig = dwt_ll_signature(img)
    zw_noisy = dwt_ll_signature(dec_noisy)
    zw_rs_orig = rs_encode(zw_orig, nsym=16)
    zw_rs_noisy = rs_encode(zw_noisy, nsym=16)
    zw_rs_dec_orig = rs_decode(zw_rs_orig, nsym=16)
    zw_rs_dec_noisy = rs_decode(zw_rs_noisy, nsym=16)
    zw_ecc_dist = hamming(zw_rs_dec_orig, zw_rs_dec_noisy)

    # ROI IWT + zlib compression measurement (lossless)
    iwt_total_bytes = ll_diff_zlib_size(LL) + rle_zero_zlib_size(LH) + rle_zero_zlib_size(HL) + rle_zero_zlib_size(HH)

    result = {
        "mask": mask_ref,
        "roi": roi,
        "bg": bg,
        "compression": comp,
        "roi_psnr": psnr_roi,
        "roi_ssim": ssim_roi,
        "enc": enc,
        "dec_noisy": dec_noisy,
        "ph_orig": ph_orig,
        "ph_noisy": ph_noisy,
        "ph_dist": dist,
        "sha_match": sha_orig == sha_noisy,
        "sha256": sha_orig,
        "npcr": npcr,
        "uaci": uaci,
        "iwt_zlib_bytes": iwt_total_bytes,
        "zw_dist": zw_ecc_dist,
    }
    _, orig_png = cv2.imencode(".png", img)
    result["orig_png_bytes"] = len(orig_png)
    result["compression_ratio"] = result["orig_png_bytes"] / result["compression"]["total"] if result["compression"]["total"] > 0 else 0
    result["enc_entropy_bits"] = entropy_bits(enc)
    result["q_bg"] = q_bg
    return result


def main():
    images = get_medmnist("chestmnist", split="test", max_items=1500)
    print(f"Loaded {len(images)} images from MedMNIST ChestMNIST")
    totals = {
        "bytes_total": 0,
        "bytes_orig_png_total": 0,
        "bytes_iwt_zlib_total": 0,
        "psnr_sum": 0.0,
        "ssim_sum": 0.0,
        "ph_dist_sum": 0,
        "sha_match_count": 0,
        "entropy_sum": 0.0,
        "cr_sum": 0.0,
        "ph_min": 1e9,
        "ph_max": -1e9,
        "npcr_sum": 0.0,
        "uaci_sum": 0.0,
        "zw_sum": 0,
        "zw_min": 1e9,
        "zw_max": -1e9,
    }
    for idx, img in enumerate(images):
        res = process_image(img)
        # Accumulate stats
        totals["bytes_total"] += res["compression"]["total"]
        totals["psnr_sum"] += res["roi_psnr"]
        totals["ssim_sum"] += res["roi_ssim"]
        totals["ph_dist_sum"] += res["ph_dist"]
        totals["sha_match_count"] += 1 if res["sha_match"] else 0
        totals["entropy_sum"] += res["enc_entropy_bits"]
        totals["cr_sum"] += res["compression_ratio"]
        totals["bytes_orig_png_total"] += res["orig_png_bytes"]
        totals["bytes_iwt_zlib_total"] += res["iwt_zlib_bytes"]
        totals["ph_min"] = min(totals["ph_min"], res["ph_dist"])
        totals["ph_max"] = max(totals["ph_max"], res["ph_dist"])
        totals["npcr_sum"] += res["npcr"]
        totals["uaci_sum"] += res["uaci"]
        totals["zw_sum"] += res["zw_dist"]
        totals["zw_min"] = min(totals["zw_min"], res["zw_dist"])
        totals["zw_max"] = max(totals["zw_max"], res["zw_dist"])

        # Save a few examples for the report
        if idx < 3:
            save_image(img, os.path.join(IMG_DIR, f"dataset_original_{idx}.png"))
            save_image(res["mask"], os.path.join(IMG_DIR, f"dataset_mask_{idx}.png"))
            save_image(res["roi"], os.path.join(IMG_DIR, f"dataset_roi_{idx}.png"))
            save_image(res["bg"], os.path.join(IMG_DIR, f"dataset_bg_{idx}.png"))
            save_image(res["enc"], os.path.join(IMG_DIR, f"dataset_enc_{idx}.png"))
            save_image(res["dec_noisy"], os.path.join(IMG_DIR, f"dataset_decnoisy_{idx}.png"))

        # On-chain test (EthereumTester): register first item
        if idx == 0:
            rpc = os.environ.get("ETH_RPC_URL")
            pkey = os.environ.get("PRIVATE_KEY")
            chain = ImageChain(ChainConfig(rpc_url=rpc, private_key=pkey))
            addr = chain.deploy()
            print(f"Deployed ImageRegistry at {addr}")
            for j in range(min(20, len(images))):
                imgj = images[j]
                sha_j = hashlib.sha256(imgj.tobytes()).hexdigest()
                ph_j = phash64(imgj)
                tx_hash = chain.register(
                    sha256_hash_hex=sha_j,
                    phash_int=ph_j,
                    patient_id=f"P-TEST-{j:03d}",
                    note="ChestMNIST batch",
                )
                exists = chain.exists(sha_j)
                print(f"Chain TX: {tx_hash}, exists={exists}")

    n = len(images)
    avg_psnr = totals["psnr_sum"] / n
    avg_ssim = totals["ssim_sum"] / n
    avg_phdist = totals["ph_dist_sum"] / n
    avg_entropy = totals["entropy_sum"] / n
    avg_cr = totals["cr_sum"] / n
    avg_npcr = totals["npcr_sum"] / n
    avg_uaci = totals["uaci_sum"] / n
    print(f"Processed {n} images.")
    print(f"Avg ROI PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
    print(f"Avg pHash Hamming Distance (noisy decryption): {avg_phdist:.2f} (min={totals['ph_min']}, max={totals['ph_max']})")
    print(f"Avg encrypted entropy: {avg_entropy:.2f} bits")
    print(f"Avg compression ratio (orig PNG / ROI+BG+Mask): {avg_cr:.2f}")
    print(f"Avg ROI IWT RLE+zlib bytes: {totals['bytes_iwt_zlib_total'] // n} (ROI PNG avg: {totals['bytes_orig_png_total'] // n})")
    print(f"Avg NPCR: {avg_npcr:.2f}% , Avg UACI: {avg_uaci:.2f}%")
    print(f"Avg Zero-Watermark Hamming Distance: {totals['zw_sum']/n:.2f} (min={totals['zw_min']}, max={totals['zw_max']})")
    print(f"SHA-256 match count (should be 0 due to noise): {totals['sha_match_count']}/{n}")


if __name__ == "__main__":
    main()

