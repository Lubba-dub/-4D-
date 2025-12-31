import numpy as np


def logistic_key(length: int, x0: float = 0.123456789, r: float = 3.9999) -> np.ndarray:
    key = np.zeros(length, dtype=np.uint8)
    x = x0
    for _ in range(100):
        x = r * x * (1 - x)
    for i in range(length):
        x = r * x * (1 - x)
        key[i] = int((x * 1000) % 256)
    return key


def xor_stream(img: np.ndarray, key: np.ndarray) -> np.ndarray:
    flat = img.flatten()
    enc = np.bitwise_xor(flat, key[: len(flat)])
    return enc.reshape(img.shape)


def logistic_sine_key(length: int, x0: float = 0.31337, y0: float = 0.73123, r: float = 3.999, a: float = 0.99) -> np.ndarray:
    """
    Logistic-Sine coupling chaotic sequence to improve randomness.
    x_{n+1} = r * x_n * (1 - x_n)
    y_{n+1} = a * sin(pi * y_n)
    key_n = floor(mod((x_{n+1} + y_{n+1}) * 10^6, 256))
    """
    key = np.zeros(length, dtype=np.uint8)
    x = x0
    y = y0
    for _ in range(200):
        x = r * x * (1 - x)
        y = a * np.sin(np.pi * y)
    for i in range(length):
        x = r * x * (1 - x)
        y = a * np.sin(np.pi * y)
        k = (x + y) * 1e6
        key[i] = int(k) % 256
    return key


def xor_diffuse_encrypt(img: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Simple diffusion: c[i] = p[i] XOR key[i] XOR c[i-1]
    """
    p = img.flatten().astype(np.uint8)
    k = key[: len(p)]
    c = np.zeros_like(p)
    prev = k[0]
    for i in range(len(p)):
        ci = p[i] ^ k[i] ^ prev
        c[i] = ci
        prev = ci
    return c.reshape(img.shape)


def xor_diffuse_decrypt(cipher: np.ndarray, key: np.ndarray) -> np.ndarray:
    c = cipher.flatten().astype(np.uint8)
    k = key[: len(c)]
    p = np.zeros_like(c)
    prev = k[0]
    for i in range(len(c)):
        pi = c[i] ^ k[i] ^ prev
        p[i] = pi
        prev = c[i]
    return p.reshape(cipher.shape)


def _permutation_from_key(length: int, key: np.ndarray) -> np.ndarray:
    # Use chaotic key to produce a permutation via argsort
    vals = key[:length].astype(np.int64)
    # Expand with a simple hash to reduce collisions
    vals = (vals * 1315423911) % (1 << 32)
    perm = np.argsort(vals, kind="mergesort")
    return perm


def perm_diffuse_encrypt(img: np.ndarray, key: np.ndarray) -> np.ndarray:
    p = img.flatten().astype(np.uint8)
    perm = _permutation_from_key(len(p), key)
    p_perm = p[perm]
    # Diffuse over permuted sequence
    k = key[: len(p_perm)]
    c_perm = np.zeros_like(p_perm)
    prev = k[0]
    for i in range(len(p_perm)):
        ci = p_perm[i] ^ k[i] ^ prev
        c_perm[i] = ci
        prev = ci
    # Un-permute ciphertext back to image order (store in array)
    inv = np.zeros_like(perm)
    inv[perm] = np.arange(len(perm))
    c = c_perm[inv]
    return c.reshape(img.shape)


def perm_diffuse_decrypt(cipher: np.ndarray, key: np.ndarray) -> np.ndarray:
    c = cipher.flatten().astype(np.uint8)
    perm = _permutation_from_key(len(c), key)
    # Apply permutation to ciphertext to get permuted sequence
    c_perm = c[perm]
    # De-diffuse over permuted sequence
    k = key[: len(c_perm)]
    p_perm = np.zeros_like(c_perm)
    prev = k[0]
    for i in range(len(c_perm)):
        pi = c_perm[i] ^ k[i] ^ prev
        p_perm[i] = pi
        prev = c_perm[i]
    # Invert permutation to original order
    inv = np.zeros_like(perm)
    inv[perm] = np.arange(len(perm))
    p = p_perm[inv]
    return p.reshape(cipher.shape)


def perm_bidiffuse_encrypt(img: np.ndarray, key1: np.ndarray, key2: np.ndarray) -> np.ndarray:
    img_u8 = img.astype(np.uint8)
    h, w = img_u8.shape
    p = img_u8.flatten()
    perm = _permutation_from_key(len(p), key1)
    p_perm = p[perm].reshape(h, w)
    # Row-wise forward/backward diffusion with dynamic key update
    k1 = key1[: h * w].reshape(h, w)
    c = np.zeros_like(p_perm)
    for r in range(h):
        prev = k1[r, 0]
        for cidx in range(w):
            k1[r, cidx] = (k1[r, cidx] + prev) % 256
            val = p_perm[r, cidx] ^ k1[r, cidx] ^ prev
            c[r, cidx] = val
            prev = val
        nextv = k1[r, w - 1]
        for cidx in range(w - 1, -1, -1):
            k1[r, cidx] = (k1[r, cidx] + nextv) % 256
            c[r, cidx] = c[r, cidx] ^ k1[r, cidx] ^ nextv
            nextv = c[r, cidx]
    # Column-wise forward/backward diffusion
    k2 = key2[: h * w].reshape(h, w)
    for cidx in range(w):
        prev = k2[0, cidx]
        for r in range(h):
            k2[r, cidx] = (k2[r, cidx] + prev) % 256
            c[r, cidx] = c[r, cidx] ^ k2[r, cidx] ^ prev
            prev = c[r, cidx]
        nextv = k2[h - 1, cidx]
        for r in range(h - 1, -1, -1):
            k2[r, cidx] = (k2[r, cidx] + nextv) % 256
            c[r, cidx] = c[r, cidx] ^ k2[r, cidx] ^ nextv
            nextv = c[r, cidx]
    # Un-permute to image order
    inv = np.zeros_like(perm)
    inv[perm] = np.arange(len(perm))
    out = c.flatten()[inv]
    return out.reshape(h, w)


def perm_bidiffuse_decrypt(cipher: np.ndarray, key1: np.ndarray, key2: np.ndarray) -> np.ndarray:
    h, w = cipher.shape
    c = cipher.flatten().astype(np.uint8)
    perm = _permutation_from_key(len(c), key1)
    # Permute ciphertext then invert column-wise and row-wise diffusion
    c_perm = c[perm].reshape(h, w)
    k2 = key2[: h * w].reshape(h, w)
    # Invert column-wise backward and forward
    for cidx in range(w):
        nextv = k2[h - 1, cidx]
        for r in range(h - 1, -1, -1):
            k2[r, cidx] = (k2[r, cidx] + nextv) % 256
            c_perm[r, cidx] = c_perm[r, cidx] ^ k2[r, cidx] ^ nextv
            nextv = c_perm[r, cidx]
        prev = k2[0, cidx]
        for r in range(h):
            k2[r, cidx] = (k2[r, cidx] + prev) % 256
            c_perm[r, cidx] = c_perm[r, cidx] ^ k2[r, cidx] ^ prev
            prev = c_perm[r, cidx]
    k1 = key1[: h * w].reshape(h, w)
    # Invert row-wise backward and forward
    for r in range(h):
        nextv = k1[r, w - 1]
        for cidx in range(w - 1, -1, -1):
            k1[r, cidx] = (k1[r, cidx] + nextv) % 256
            c_perm[r, cidx] = c_perm[r, cidx] ^ k1[r, cidx] ^ nextv
            nextv = c_perm[r, cidx]
        prev = k1[r, 0]
        for cidx in range(w):
            k1[r, cidx] = (k1[r, cidx] + prev) % 256
            c_perm[r, cidx] = c_perm[r, cidx] ^ k1[r, cidx] ^ prev
            prev = c_perm[r, cidx]
    # Invert permutation
    inv = np.zeros_like(perm)
    inv[perm] = np.arange(len(perm))
    p = c_perm.flatten()[inv]
    return p.reshape(h, w)

