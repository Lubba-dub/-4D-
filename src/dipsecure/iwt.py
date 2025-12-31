import numpy as np


def cdf53_forward_1d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.int32)
    n = len(x)
    if n % 2 != 0:
        x = np.append(x, x[-1])
        n += 1
    s = x[0::2].copy()
    d = x[1::2].copy()
    s_next = np.roll(s, -1)
    s_next[-1] = s[-1]
    d -= np.floor((s + s_next) / 2).astype(np.int32)
    d_prev = np.roll(d, 1)
    d_prev[0] = d[0]
    s += np.floor((d_prev + d + 2) / 4).astype(np.int32)
    return s, d


def cdf53_inverse_1d(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    s = s.astype(np.int32)
    d = d.astype(np.int32)
    d_prev = np.roll(d, 1)
    d_prev[0] = d[0]
    s -= np.floor((d_prev + d + 2) / 4).astype(np.int32)
    s_next = np.roll(s, -1)
    s_next[-1] = s[-1]
    d += np.floor((s + s_next) / 2).astype(np.int32)
    x_rec = np.zeros(2 * len(s), dtype=np.int32)
    x_rec[0::2] = s
    x_rec[1::2] = d
    return x_rec


def iwt2(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = img.shape
    rows -= rows % 2
    cols -= cols % 2
    img = img[:rows, :cols].astype(np.int32)

    L = np.zeros((rows, cols // 2), dtype=np.int32)
    H = np.zeros((rows, cols // 2), dtype=np.int32)
    for i in range(rows):
        s, d = cdf53_forward_1d(img[i, :])
        L[i, :] = s
        H[i, :] = d
    LL = np.zeros((rows // 2, cols // 2), dtype=np.int32)
    LH = np.zeros((rows // 2, cols // 2), dtype=np.int32)
    HL = np.zeros((rows // 2, cols // 2), dtype=np.int32)
    HH = np.zeros((rows // 2, cols // 2), dtype=np.int32)
    for j in range(cols // 2):
        s, d = cdf53_forward_1d(L[:, j])
        LL[:, j] = s
        LH[:, j] = d
        s, d = cdf53_forward_1d(H[:, j])
        HL[:, j] = s
        HH[:, j] = d
    return LL, LH, HL, HH


def iiwt2(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray) -> np.ndarray:
    rows_half, cols_half = LL.shape
    rows = rows_half * 2
    cols = cols_half * 2
    L_rec = np.zeros((rows, cols_half), dtype=np.int32)
    H_rec = np.zeros((rows, cols_half), dtype=np.int32)
    for j in range(cols_half):
        L_rec[:, j] = cdf53_inverse_1d(LL[:, j], LH[:, j])
        H_rec[:, j] = cdf53_inverse_1d(HL[:, j], HH[:, j])
    img_rec = np.zeros((rows, cols), dtype=np.int32)
    for i in range(rows):
        img_rec[i, :] = cdf53_inverse_1d(L_rec[i, :], H_rec[i, :])
    return img_rec

