import numpy as np
import cv2
import hashlib


def init_dct_dictionary(bs: int = 8) -> np.ndarray:
    n = bs * bs
    D = np.zeros((n, n), dtype=np.float32)
    idx = 0
    for u in range(bs):
        for v in range(bs):
            basis = np.zeros((bs, bs), dtype=np.float32)
            basis[u, v] = 1.0
            atom = cv2.idct(basis)
            D[:, idx] = atom.flatten()
            idx += 1
    for i in range(n):
        norm = np.linalg.norm(D[:, i]) + 1e-6
        D[:, i] /= norm
    return D


def keyed_transform(D: np.ndarray, seed: str) -> np.ndarray:
    h = hashlib.sha256(seed.encode("utf-8")).digest()
    rng = np.random.default_rng(np.frombuffer(h, dtype=np.uint64))
    n = D.shape[1]
    R = rng.standard_normal((n, n)).astype(np.float32)
    Q, _ = np.linalg.qr(R)
    T = D @ Q
    for i in range(n):
        norm = np.linalg.norm(T[:, i]) + 1e-6
        T[:, i] /= norm
    return T


def omp_codes(D: np.ndarray, patch: np.ndarray, k: int) -> np.ndarray:
    x = patch.flatten().astype(np.float32)
    residual = x.copy()
    idxs = []
    A = np.empty((D.shape[0], 0), dtype=np.float32)
    for _ in range(k):
        correlations = D.T @ residual
        j = int(np.argmax(np.abs(correlations)))
        if j in idxs:
            break
        idxs.append(j)
        A = D[:, idxs]
        c, *_ = np.linalg.lstsq(A, x, rcond=None)
        residual = x - A @ c
        if np.linalg.norm(residual) < 1e-3:
            break
    codes = np.zeros(D.shape[1], dtype=np.float32)
    codes[idxs] = c.astype(np.float32)
    return codes


def reconstruct_patch(D: np.ndarray, codes: np.ndarray, bs: int = 8) -> np.ndarray:
    x = (D @ codes).reshape(bs, bs)
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def pca_dictionary(patches: np.ndarray, bs: int = 8) -> np.ndarray:
    n = bs * bs
    X = patches.reshape(patches.shape[0], n).astype(np.float32)
    X = X - np.mean(X, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    D = Vt[:n, :].T
    for i in range(n):
        norm = np.linalg.norm(D[:, i]) + 1e-6
        D[:, i] /= norm
    return D.astype(np.float32)


def ksvd_train(D: np.ndarray, patches: np.ndarray, bs: int = 8, k: int = 10, iters: int = 3) -> np.ndarray:
    n = bs * bs
    X = patches.reshape(patches.shape[0], n).astype(np.float32)
    for _ in range(iters):
        codes_list = []
        for p in patches:
            codes = omp_codes(D, p, k)
            codes_list.append(codes)
        C = np.stack(codes_list, axis=1)
        for j in range(n):
            idx = np.where(np.abs(C[j, :]) > 1e-6)[0]
            if len(idx) == 0:
                continue
            Ej = X.T - D @ C + np.outer(D[:, j], C[j, :])
            E_sub = Ej[:, idx]
            U, S, Vt = np.linalg.svd(E_sub, full_matrices=False)
            D[:, j] = U[:, 0]
            C[j, idx] = S[0] * Vt[0, :]
        for i in range(n):
            norm = np.linalg.norm(D[:, i]) + 1e-6
            D[:, i] /= norm
    return D.astype(np.float32)
