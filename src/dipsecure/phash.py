import numpy as np
import cv2


def phash64(img: np.ndarray) -> int:
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[0:8, 0:8]
    avg = np.mean(low)
    h = 0
    for i in range(8):
        for j in range(8):
            h <<= 1
            if low[i, j] > avg:
                h |= 1
    return h


def hamming(a: int, b: int) -> int:
    x = a ^ b
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c

