import numpy as np
import cv2


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 * img1, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compression_bytes(roi: np.ndarray, background: np.ndarray, mask: np.ndarray, q_bg: int = 40) -> dict:
    result, roi_png = cv2.imencode(".png", roi)
    result2, bg_jpg = cv2.imencode(".jpg", background, [int(cv2.IMWRITE_JPEG_QUALITY), q_bg])
    result3, mask_png = cv2.imencode(".png", mask)
    return {
        "roi_png": len(roi_png),
        "bg_jpg": len(bg_jpg),
        "mask_png": len(mask_png),
        "total": len(roi_png) + len(bg_jpg) + len(mask_png),
    }


def entropy_bits(img: np.ndarray) -> float:
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    p = hist / np.sum(hist)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def edge_activity(img: np.ndarray) -> float:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag))


def adaptive_bg_quality(img_bg: np.ndarray) -> int:
    a = edge_activity(img_bg)
    if a < 5.0:
        return 30
    elif a < 15.0:
        return 40
    else:
        return 50


def npcr_uaci(img_enc_a: np.ndarray, img_enc_b: np.ndarray) -> tuple[float, float]:
    """
    Compute NPCR and UACI between two encrypted images.
    NPCR: Number of Pixels Change Rate
    UACI: Unified Average Changing Intensity
    """
    A = img_enc_a.astype(np.int32)
    B = img_enc_b.astype(np.int32)
    H, W = A.shape
    D = (A != B).astype(np.int32)
    npcr = 100.0 * np.sum(D) / (H * W)
    uaci = 100.0 * np.mean(np.abs(A - B) / 255.0)
    return float(npcr), float(uaci)


def zlib_compress_size(arr: np.ndarray, level: int = 9) -> int:
    import zlib
    data = arr.astype(np.int32).tobytes()
    comp = zlib.compress(data, level)
    return len(comp)


def rle_zero_zlib_size(arr: np.ndarray, level: int = 9) -> int:
    """
    Very simple RLE for zeros:
    Encode as pairs (value, countZeroRunAfterValue).
    """
    import zlib
    flat = arr.flatten().astype(np.int32)
    out = []
    i = 0
    n = len(flat)
    while i < n:
        v = flat[i]
        j = i + 1
        count = 0
        while j < n and flat[j] == 0:
            count += 1
            j += 1
        out.append(v)
        out.append(count)
        i = j
    data = np.array(out, dtype=np.int32).tobytes()
    comp = zlib.compress(data, level)
    return len(comp)


def ll_diff_zlib_size(arr: np.ndarray, level: int = 9) -> int:
    """
    Row-wise first-order differencing for LL band, then zlib.
    """
    import zlib
    a = arr.astype(np.int32)
    diff = np.zeros_like(a)
    diff[:, 0] = a[:, 0]
    diff[:, 1:] = a[:, 1:] - a[:, :-1]
    data = diff.flatten().tobytes()
    comp = zlib.compress(data, level)
    return len(comp)

