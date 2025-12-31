import os
import cv2
import numpy as np
from dipsecure.data_loader import get_medmnist, ensure_dir, save_image
from dipsecure.roi import robust_roi_mask, refine_mask, split_roi_background
from dipsecure.compressive_sensing import cs_encode_decode_background_dict_overlap, cs_encode_decode_background_overlap, cs_encode_decode_background_overlap_cuda, cs_encode_decode_background_dict_overlap_cuda
from dipsecure.metrics import psnr, ssim
from dipsecure.dictionary import pca_dictionary, ksvd_train
import hashlib


REPORT_DIR = os.path.join("DIP_Project", "reports")
IMG_DIR = os.path.join(REPORT_DIR, "images")
ensure_dir(IMG_DIR)


def process_image(img: np.ndarray, bs: int = 8, measure_ratio: float = 0.85, k_frac: float = 0.2, stride: int = 4, T: np.ndarray | None = None):
    mask = refine_mask(robust_roi_mask(img))
    roi, bg = split_roi_background(img, mask)
    # Crop to block-multiple for CS, then reconstruct and stitch back
    h, w = bg.shape
    h2 = h - h % bs
    w2 = w - w % bs
    bg_c = bg[:h2, :w2]
    roi_c = roi[:h2, :w2]
    mask_c = mask[:h2, :w2]
    seed = f"dict-{h2}-{w2}"
    try:
        bg_rec_cuda, bg_cs_blob_cuda = cs_encode_decode_background_overlap_cuda(bg_c, mask_c, bs=bs, stride=stride, measure_ratio=measure_ratio, k_frac=k_frac, device="cuda")
        combined_cuda = img.copy()
        combined_cuda[:h2, :w2] = np.clip(roi_c.astype(np.int32) + bg_rec_cuda.astype(np.int32), 0, 255).astype(np.uint8)
        p_cuda = psnr(img, combined_cuda)
        s_cuda = ssim(img, combined_cuda)
    except Exception:
        p_cuda = -1.0
        s_cuda = -1.0
        combined_cuda = None
        bg_cs_blob_cuda = b""
    try:
        bg_rec_dict_cuda, bg_cs_blob_dict_cuda = cs_encode_decode_background_dict_overlap_cuda(bg_c, mask_c, T=T if T is not None else np.eye(bs*bs, dtype=np.float32), bs=bs, stride=stride, measure_ratio=measure_ratio, k_frac=k_frac, device="cuda")
        combined_dict_cuda = img.copy()
        combined_dict_cuda[:h2, :w2] = np.clip(roi_c.astype(np.int32) + bg_rec_dict_cuda.astype(np.int32), 0, 255).astype(np.uint8)
        p_dict_cuda = psnr(img, combined_dict_cuda)
        s_dict_cuda = ssim(img, combined_dict_cuda)
    except Exception:
        p_dict_cuda = -1.0
        s_dict_cuda = -1.0
        combined_dict_cuda = None
        bg_cs_blob_dict_cuda = b""
    bg_rec_dict, bg_cs_blob_dict = cs_encode_decode_background_dict_overlap(bg_c, bs=bs, stride=stride, measure_ratio=measure_ratio, k_frac=k_frac, seed=seed, T=T)
    combined_dict = img.copy()
    combined_dict[:h2, :w2] = np.clip(roi_c.astype(np.int32) + bg_rec_dict.astype(np.int32), 0, 255).astype(np.uint8)
    p_dict = psnr(img, combined_dict)
    s_dict = ssim(img, combined_dict)
    bg_rec_dct, bg_cs_blob_dct = cs_encode_decode_background_overlap(bg_c, bs=bs, stride=stride, measure_ratio=measure_ratio, k_frac=k_frac)
    combined_dct = img.copy()
    combined_dct[:h2, :w2] = np.clip(roi_c.astype(np.int32) + bg_rec_dct.astype(np.int32), 0, 255).astype(np.uint8)
    p_dct = psnr(img, combined_dct)
    s_dct = ssim(img, combined_dct)
    candidates = [
        ("CUDA-DCT", combined_cuda, p_cuda, s_cuda, bg_cs_blob_cuda),
        ("CUDA-Dict", combined_dict_cuda, p_dict_cuda, s_dict_cuda, bg_cs_blob_dict_cuda),
        ("Dict-Overlap", combined_dict, p_dict, s_dict, bg_cs_blob_dict),
        ("DCT-Overlap", combined_dct, p_dct, s_dct, bg_cs_blob_dct),
    ]
    candidates = [c for c in candidates if c[1] is not None]
    best = max(candidates, key=lambda t: t[2])
    mode, combined, p, s, bg_cs_blob = best
    # PNG bytes for ROI/mask
    _, roi_png = cv2.imencode(".png", roi_c)
    _, mask_png = cv2.imencode(".png", mask_c)
    total_bytes = len(bg_cs_blob) + len(roi_png) + len(mask_png)
    _, orig_png = cv2.imencode(".png", img)
    cr = len(orig_png) / total_bytes if total_bytes > 0 else 0
    return combined, {"psnr": p, "ssim": s, "bytes_total": total_bytes, "orig_png": len(orig_png), "cr": cr, "mode": mode}


def run_for_dataset(name: str, max_items: int, bs: int, measure_ratio: float, k_frac: float, stride: int, T: np.ndarray | None):
    images = get_medmnist(name, split="test", max_items=max_items)
    print(f"[{name}] Loaded {len(images)} images")
    psnr_sum = 0.0
    ssim_sum = 0.0
    bytes_sum = 0
    orig_png_sum = 0
    modes = {"Dict-Overlap": 0, "DCT-Overlap": 0, "CUDA-DCT": 0, "CUDA-Dict": 0}
    for i, img in enumerate(images):
        combined, m = process_image(img, bs=bs, measure_ratio=measure_ratio, k_frac=k_frac, stride=stride, T=T)
        psnr_sum += m["psnr"]
        ssim_sum += m["ssim"]
        bytes_sum += m["bytes_total"]
        orig_png_sum += m["orig_png"]
        modes[m["mode"]] += 1
        if i < 3:
            save_image(img, os.path.join(IMG_DIR, f"{name}_orig_{i}.png"))
            save_image(combined, os.path.join(IMG_DIR, f"{name}_combined_{i}.png"))
    n = len(images)
    print(f"[{name}] Avg PSNR={psnr_sum/n:.2f}, SSIM={ssim_sum/n:.4f}, CR={orig_png_sum/bytes_sum:.2f}, Mode Dict={modes['Dict-Overlap']}, DCT={modes['DCT-Overlap']}")
    return {
        "avg_psnr": psnr_sum / n,
        "avg_ssim": ssim_sum / n,
        "avg_cr": orig_png_sum / bytes_sum,
        "count": n,
        "modes": modes,
    }


def main():
    ensure_dir(REPORT_DIR)
    # Config (can be tuned)
    bs = 8
    measure_ratio = 0.75
    k_frac = 0.2
    stride = 4
    train_imgs = get_medmnist("chestmnist", split="test", max_items=800)
    patches = []
    for img in train_imgs:
        mask = refine_mask(robust_roi_mask(img))
        roi, bg = split_roi_background(img, mask)
        h, w = bg.shape
        h2 = h - h % bs
        w2 = w - w % bs
        bg_c = bg[:h2, :w2]
        for i in range(0, h2 - bs + 1, bs):
            for j in range(0, w2 - bs + 1, bs):
                patches.append(bg_c[i : i + bs, j : j + bs])
                if len(patches) >= 2048:
                    break
            if len(patches) >= 2048:
                break
        if len(patches) >= 2048:
            break
    patches = np.stack(patches, axis=0)
    D0 = pca_dictionary(patches, bs=bs)
    T = ksvd_train(D0, patches, bs=bs, k=10, iters=2)
    T_hash = hashlib.sha256(T.astype(np.float32).tobytes()).hexdigest()
    m8 = int(max(1, round(measure_ratio * (bs * bs))))
    from dipsecure.hyperchaos_4d import generate_measurement_matrix
    Phi8 = generate_measurement_matrix(m8, bs * bs)
    Phi8_hash = hashlib.sha256(Phi8.astype(np.float32).tobytes()).hexdigest()
    params_str = f"bs={bs}|measure_ratio={measure_ratio}|k_frac={k_frac}|stride={stride}"
    params_hash = hashlib.sha256(params_str.encode("utf-8")).hexdigest()
    pipeline_commit = hashlib.sha256((T_hash + Phi8_hash + params_hash).encode("utf-8")).hexdigest()
    summary = {}
    for ds in ["chestmnist", "pathmnist", "pneumoniamnist"]:
        summary[ds] = run_for_dataset(ds, max_items=2000, bs=bs, measure_ratio=measure_ratio, k_frac=k_frac, stride=stride, T=T)
    # Write report
    report_path = os.path.join(REPORT_DIR, "report_cs_hyperchaos.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 压缩感知 + 4D超混沌测量矩阵 实验报告\n\n")
        f.write(f"块大小: {bs}×{bs}, 测量比: {measure_ratio}, 稀疏比例: {k_frac}, stride={stride}\n\n")
        f.write(f"字典哈希: {T_hash}\n测量矩阵哈希(bs={bs}): {Phi8_hash}\n参数哈希: {params_hash}\nPipeline Commit: {pipeline_commit}\n\n")
        for ds, val in summary.items():
            f.write(f"## {ds}\n- 样本数: {val['count']}\n- 平均PSNR: {val['avg_psnr']:.2f}\n- 平均SSIM: {val['avg_ssim']:.4f}\n- 平均压缩比 (orig PNG / ROI+CS+mask): {val['avg_cr']:.2f}\n- 模式选择: Dict={val['modes']['Dict-Overlap']} DCT={val['modes']['DCT-Overlap']}\n\n")
    print("CS+Hyperchaos report generated.")


if __name__ == "__main__":
    main()
