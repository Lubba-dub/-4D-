import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from dipsecure.data_loader import get_medmnist, ensure_dir
from dipsecure.roi import robust_roi_mask, refine_mask, split_roi_background
from dipsecure.compressive_sensing import cs_encode_decode_background_overlap, cs_encode_decode_background_dict_overlap
from dipsecure.metrics import psnr, ssim
from dipsecure.dictionary import pca_dictionary

REPORT_DIR = os.path.join("DIP_Project", "reports")
IMG_DIR = os.path.join(REPORT_DIR, "images")
ensure_dir(IMG_DIR)

def process_one(img, bs, measure_ratio, k_frac, stride, zoom_size, T=None):
    mask = refine_mask(robust_roi_mask(img))
    roi, bg = split_roi_background(img, mask)
    h, w = bg.shape
    h2 = h - h % bs
    w2 = w - w % bs
    bg_c = bg[:h2, :w2]
    roi_c = roi[:h2, :w2]
    mask_c = mask[:h2, :w2]
    bg_rec_dict, bg_blob_dict = cs_encode_decode_background_dict_overlap(bg_c, bs=bs, stride=stride, measure_ratio=measure_ratio, k_frac=k_frac, seed=f"dict-{h2}-{w2}", T=T)
    combined_dict = img.copy()
    combined_dict[:h2, :w2] = np.clip(roi_c.astype(np.int32) + bg_rec_dict.astype(np.int32), 0, 255).astype(np.uint8)
    p_dict = psnr(img, combined_dict)
    s_dict = ssim(img, combined_dict)
    bg_rec_dct, bg_blob_dct = cs_encode_decode_background_overlap(bg_c, bs=bs, stride=stride, measure_ratio=measure_ratio, k_frac=k_frac)
    combined_dct = img.copy()
    combined_dct[:h2, :w2] = np.clip(roi_c.astype(np.int32) + bg_rec_dct.astype(np.int32), 0, 255).astype(np.uint8)
    p_dct = psnr(img, combined_dct)
    s_dct = ssim(img, combined_dct)
    if p_dct > p_dict:
        combined = combined_dct
        bg_blob = bg_blob_dct
        mode = "DCT-Overlap"
        p = p_dct
        s = s_dct
    else:
        combined = combined_dict
        bg_blob = bg_blob_dict
        mode = "Dict-Overlap"
        p = p_dict
        s = s_dict
    diff = np.abs(img.astype(np.int32) - combined.astype(np.int32)).astype(np.uint8)
    z = zoom_size
    cy, cx = h2 // 2, w2 // 2
    y0 = max(0, cy - z // 2)
    x0 = max(0, cx - z // 2)
    y1 = min(h2, y0 + z)
    x1 = min(w2, x0 + z)
    zoom_o = img[y0:y1, x0:x1]
    zoom_c = combined[y0:y1, x0:x1]
    _, roi_png = cv2.imencode(".png", roi_c)
    _, mask_png = cv2.imencode(".png", mask_c)
    _, orig_png = cv2.imencode(".png", img)
    total_bytes = len(bg_blob) + len(roi_png) + len(mask_png)
    cr = len(orig_png) / total_bytes if total_bytes > 0 else 0
    hist_o, _ = np.histogram(img.flatten(), bins=256, range=(0, 255))
    hist_c, _ = np.histogram(combined.flatten(), bins=256, range=(0, 255))
    spec_o = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))))
    spec_c = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(combined.astype(np.float32)))))
    return {
        "orig": img,
        "mask": mask,
        "roi": roi,
        "bg_c": bg_c,
        "bg_rec": combined[:h2, :w2] - roi_c,
        "combined": combined,
        "diff": diff,
        "psnr": p,
        "ssim": s,
        "cr": cr,
        "mode": mode,
        "zoom_orig": zoom_o,
        "zoom_combined": zoom_c,
        "hist_o": hist_o,
        "hist_c": hist_c,
        "spec_o": spec_o,
        "spec_c": spec_c,
    }

def make_matrix(dataset, samples, bs, measure_ratio, k_frac, stride, dict_patches, zoom_size):
    imgs = get_medmnist(dataset, split="test", max_items=samples)
    patches = []
    for img in imgs:
        mask = refine_mask(robust_roi_mask(img))
        roi, bg = split_roi_background(img, mask)
        h, w = bg.shape
        h2 = h - h % bs
        w2 = w - w % bs
        bg_c = bg[:h2, :w2]
        for i in range(0, h2 - bs + 1, bs):
            for j in range(0, w2 - bs + 1, bs):
                patches.append(bg_c[i : i + bs, j : j + bs])
                if len(patches) >= dict_patches:
                    break
            if len(patches) >= dict_patches:
                break
        if len(patches) >= dict_patches:
            break
    T = pca_dictionary(np.stack(patches, axis=0), bs=bs) if patches else None
    rows = []
    for img in imgs:
        rows.append(process_one(img, bs, measure_ratio, k_frac, stride, zoom_size, T=T))
    cols = ["orig", "mask", "roi", "bg_c", "bg_rec", "combined", "diff", "zoom_orig", "zoom_combined", "hist", "spec_orig", "spec_combined"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(len(cols)*3, len(rows)*3))
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)
    mappable = None
    for i, row in enumerate(rows):
        for j, key in enumerate(cols):
            ax = axes[i][j]
            ax.axis("off")
            if key == "diff":
                im = ax.imshow(row[key], cmap="inferno")
                mappable = im
            elif key == "hist":
                ax.plot(np.arange(256), row["hist_o"], color="blue", linewidth=0.7)
                ax.plot(np.arange(256), row["hist_c"], color="red", linewidth=0.7)
            elif key == "spec_orig":
                ax.imshow(row["spec_o"], cmap="magma")
            elif key == "spec_combined":
                ax.imshow(row["spec_c"], cmap="magma")
            else:
                ax.imshow(row[key], cmap="gray")
            if key == "combined":
                ax.set_title(f"{row['mode']} | PSNR {row['psnr']:.2f} SSIM {row['ssim']:.3f} CR {row['cr']:.2f}", fontsize=8)
            else:
                ax.set_title(key, fontsize=8)
    plt.tight_layout()
    if mappable is not None:
        try:
            plt.colorbar(mappable, ax=axes[:, -1], shrink=0.6)
        except Exception:
            pass
    out_path = os.path.join(IMG_DIR, f"cs_matrix_{dataset}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["chestmnist", "pathmnist", "pneumoniamnist"])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--measure_ratio", type=float, default=0.85)
    parser.add_argument("--k_frac", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--dict_patches", type=int, default=256)
    parser.add_argument("--zoom", type=int, default=16)
    args = parser.parse_args()
    ensure_dir(REPORT_DIR)
    paths = []
    for ds in args.datasets:
        p = make_matrix(ds, samples=args.samples, bs=args.bs, measure_ratio=args.measure_ratio, k_frac=args.k_frac, stride=args.stride, dict_patches=args.dict_patches, zoom_size=args.zoom)
        paths.append((ds, p))
    report_path = os.path.join(REPORT_DIR, "report_cs_visual_matrix.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 压缩后-解压可视化矩阵报告\n\n")
        f.write(f"参数: bs={args.bs}, measure_ratio={args.measure_ratio}, k_frac={args.k_frac}, stride={args.stride}, dict_patches={args.dict_patches}, zoom={args.zoom}\n\n")
        f.write("列含义: orig, mask, roi, bg_c, bg_rec, combined, diff, zoom_orig, zoom_combined, hist(蓝=orig,红=combined), spec_orig, spec_combined\n\n")
        for ds, p in paths:
            rel = os.path.relpath(p, REPORT_DIR).replace("\\", "/")
            f.write(f"## {ds}\n")
            f.write(f"![{ds} matrix]({rel})\n\n")
    print("Visual matrix report generated.")

if __name__ == "__main__":
    main()
