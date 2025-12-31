import numpy as np
import cv2
from typing import Tuple
from .hyperchaos_4d import generate_measurement_matrix
from .dictionary import init_dct_dictionary, keyed_transform, omp_codes, reconstruct_patch
from .dictionary import ksvd_train


def dct2(block: np.ndarray) -> np.ndarray:
    return cv2.dct(block.astype(np.float32))


def idct2(coeff: np.ndarray) -> np.ndarray:
    return cv2.idct(coeff.astype(np.float32))


def blockify(img: np.ndarray, bs: int = 8) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape
    h2 = h - h % bs
    w2 = w - w % bs
    img = img[:h2, :w2]
    blocks = img.reshape(h2 // bs, bs, w2 // bs, bs).swapaxes(1, 2).reshape(-1, bs, bs)
    return blocks, h2, w2


def unblockify(blocks: np.ndarray, h: int, w: int, bs: int = 8) -> np.ndarray:
    out = blocks.reshape(h // bs, w // bs, bs, bs).swapaxes(1, 2).reshape(h, w)
    return out


def hard_threshold(coeff: np.ndarray, k: int) -> np.ndarray:
    flat = coeff.flatten()
    idx = np.argsort(-np.abs(flat))[:k]
    x = np.zeros_like(flat)
    x[idx] = flat[idx]
    return x


def omp(Phi: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    m, n = Phi.shape
    residual = y.copy().astype(np.float32)
    idxs = []
    A = np.empty((m, 0), dtype=np.float32)
    for _ in range(k):
        correlations = Phi.T @ residual
        j = int(np.argmax(np.abs(correlations)))
        if j in idxs:
            break
        idxs.append(j)
        A = Phi[:, idxs]
        # Least squares
        x_ls, *_ = np.linalg.lstsq(A, y, rcond=None)
        residual = y - A @ x_ls
        if np.linalg.norm(residual) < 1e-3:
            break
    x_hat = np.zeros(n, dtype=np.float32)
    x_hat[idxs] = x_ls.astype(np.float32)
    return x_hat


def cs_encode_decode_background(bg: np.ndarray, bs: int = 8, measure_ratio: float = 0.5, k_frac: float = 0.1) -> Tuple[np.ndarray, bytes]:
    blocks, h2, w2 = blockify(bg, bs=bs)
    n = bs * bs
    m = int(max(1, round(measure_ratio * n)))
    Phi = generate_measurement_matrix(m, n)
    recon_blocks = []
    Y_list = []
    k = max(1, int(round(k_frac * n)))
    for b in blocks:
        coeff = dct2(b)
        x = hard_threshold(coeff, k)  # sparse vector length n
        y = Phi @ x.astype(np.float32)
        x_hat = omp(Phi, y, k)
        coeff_hat = x_hat.reshape(bs, bs)
        b_rec = idct2(coeff_hat)
        b_rec = np.clip(b_rec, 0, 255)
        recon_blocks.append(b_rec.astype(np.uint8))
        Y_list.append(y.astype(np.float32).tobytes())
    recon = unblockify(np.stack(recon_blocks, axis=0), h2, w2, bs=bs)
    # Compress measurements using zlib
    import zlib

    blob = b"".join(Y_list)
    blob_comp = zlib.compress(blob, 9)
    return recon, blob_comp


def cs_encode_decode_background_overlap(bg: np.ndarray, bs: int = 8, stride: int = 4, measure_ratio: float = 0.7, k_frac: float = 0.2, noise_std: float = 0.0) -> Tuple[np.ndarray, bytes]:
    h, w = bg.shape
    h2 = h - (h - bs) % stride
    w2 = w - (w - bs) % stride
    canvas = np.zeros((h2, w2), dtype=np.float32)
    weight = np.zeros((h2, w2), dtype=np.float32)
    n = bs * bs
    m = int(max(1, round(measure_ratio * n)))
    Phi = generate_measurement_matrix(m, n)
    k = max(1, int(round(k_frac * n)))
    Y_list = []
    for i in range(0, h2 - bs + 1, stride):
        for j in range(0, w2 - bs + 1, stride):
            b = bg[i : i + bs, j : j + bs]
            coeff = dct2(b)
            x = hard_threshold(coeff, k)
            y = Phi @ x.astype(np.float32)
            if noise_std > 0:
                y += np.random.normal(0, noise_std, y.shape).astype(np.float32)
            x_hat = omp(Phi, y, k)
            coeff_hat = x_hat.reshape(bs, bs)
            b_rec = idct2(coeff_hat)
            b_rec = np.clip(b_rec, 0, 255).astype(np.float32)
            canvas[i : i + bs, j : j + bs] += b_rec
            weight[i : i + bs, j : j + bs] += 1.0
            Y_list.append(y.astype(np.float32).tobytes())
    recon = canvas / np.maximum(weight, 1e-6)
    recon = cv2.bilateralFilter(recon.astype(np.uint8), d=5, sigmaColor=25, sigmaSpace=25)
    import zlib
    blob = b"".join(Y_list)
    blob_comp = zlib.compress(blob, 9)
    return recon, blob_comp


def cs_encode_decode_background_dict_overlap(bg: np.ndarray, bs: int = 8, stride: int = 4, measure_ratio: float = 0.85, k_frac: float = 0.2, seed: str = "dict-seed", T: np.ndarray | None = None, noise_std: float = 0.0) -> Tuple[np.ndarray, bytes]:
    h, w = bg.shape
    h2 = h - (h - bs) % stride
    w2 = w - (w - bs) % stride
    canvas = np.zeros((h2, w2), dtype=np.float32)
    weight = np.zeros((h2, w2), dtype=np.float32)
    n = bs * bs
    m = int(max(1, round(measure_ratio * n)))
    Phi = generate_measurement_matrix(m, n)
    if T is None:
        D = init_dct_dictionary(bs)
        T = keyed_transform(D, seed)
    k = max(1, int(round(k_frac * n)))
    Y_list = []
    for i in range(0, h2 - bs + 1, stride):
        for j in range(0, w2 - bs + 1, stride):
            b = bg[i : i + bs, j : j + bs]
            codes = omp_codes(T, b, k)
            y = Phi @ codes.astype(np.float32)
            if noise_std > 0:
                y += np.random.normal(0, noise_std, y.shape).astype(np.float32)
            codes_hat = omp(Phi, y, k)
            b_rec = reconstruct_patch(T, codes_hat, bs=bs).astype(np.float32)
            canvas[i : i + bs, j : j + bs] += b_rec
            weight[i : i + bs, j : j + bs] += 1.0
            Y_list.append(y.astype(np.float32).tobytes())
    recon = canvas / np.maximum(weight, 1e-6)
    recon = cv2.bilateralFilter(recon.astype(np.uint8), d=5, sigmaColor=25, sigmaSpace=25)
    import zlib
    blob = b"".join(Y_list)
    blob_comp = zlib.compress(blob, 9)
    return recon, blob_comp


def cs_encode_decode_background_overlap_cuda(bg: np.ndarray, mask: np.ndarray, bs: int = 8, stride: int = 4, measure_ratio: float = 0.8, k_frac: float = 0.2, device: str = "cuda", noise_std: float = 0.0) -> Tuple[np.ndarray, bytes]:
    try:
        import torch
    except Exception:
        return cs_encode_decode_background_overlap(bg, bs=bs, stride=stride, measure_ratio=measure_ratio, k_frac=k_frac, noise_std=noise_std)
    if not torch.cuda.is_available() and device == "cuda":
        return cs_encode_decode_background_overlap(bg, bs=bs, stride=stride, measure_ratio=measure_ratio, k_frac=k_frac, noise_std=noise_std)
    h, w = bg.shape
    h2 = h - (h - bs) % stride
    w2 = w - (w - bs) % stride
    canvas = np.zeros((h2, w2), dtype=np.float32)
    weight = np.zeros((h2, w2), dtype=np.float32)
    n = bs * bs
    m = int(max(1, round(measure_ratio * n)))
    Phi_np = generate_measurement_matrix(m, n)
    Phi = torch.from_numpy(Phi_np).to(device)
    k = max(1, int(round(k_frac * n)))
    Y_list = []
    for i in range(0, h2 - bs + 1, stride):
        for j in range(0, w2 - bs + 1, stride):
            mb = mask[i : i + bs, j : j + bs]
            roi_ratio = float(np.count_nonzero(mb)) / float(bs * bs * 255) if np.max(mb) > 0 else 0.0
            if roi_ratio < 0.2:
                b = bg[i : i + bs, j : j + bs]
                coeff = dct2(b)
                x = hard_threshold(coeff, k)
                x_t = torch.from_numpy(x.astype(np.float32)).to(device)
                y = Phi @ x_t
                if noise_std > 0:
                    y += torch.randn_like(y) * noise_std
                residual = y.clone()
                idxs = []
                A = None
                for _ in range(k):
                    corr = torch.matmul(Phi.T, residual)
                    jmax = int(torch.argmax(torch.abs(corr)).item())
                    if jmax in idxs:
                        break
                    idxs.append(jmax)
                    A = Phi[:, idxs]
                    sol = torch.linalg.lstsq(A, y).solution
                    residual = y - A @ sol
                    if torch.norm(residual).item() < 1e-3:
                        break
                x_hat = torch.zeros(n, device=device, dtype=torch.float32)
                if len(idxs) > 0:
                    x_hat[idxs] = sol
                x_cpu = x_hat.detach().cpu().numpy().reshape(bs, bs)
                b_rec = idct2(x_cpu)
                b_rec = np.clip(b_rec, 0, 255).astype(np.float32)
                canvas[i : i + bs, j : j + bs] += b_rec
                weight[i : i + bs, j : j + bs] += 1.0
                Y_list.append(y.detach().cpu().numpy().astype(np.float32).tobytes())
            else:
                b = bg[i : i + bs, j : j + bs].astype(np.float32)
                canvas[i : i + bs, j : j + bs] += b
                weight[i : i + bs, j : j + bs] += 1.0
    recon = canvas / np.maximum(weight, 1e-6)
    recon = cv2.bilateralFilter(recon.astype(np.uint8), d=5, sigmaColor=25, sigmaSpace=25)
    import zlib
    blob = b"".join(Y_list)
    blob_comp = zlib.compress(blob, 9)
    return recon, blob_comp


def cs_encode_decode_background_dict_overlap_cuda(bg: np.ndarray, mask: np.ndarray, T: np.ndarray, bs: int = 8, stride: int = 4, measure_ratio: float = 0.85, k_frac: float = 0.2, device: str = "cuda", noise_std: float = 0.0) -> Tuple[np.ndarray, bytes]:
    try:
        import torch
    except Exception:
        return cs_encode_decode_background_dict_overlap(bg, bs=bs, stride=stride, measure_ratio=measure_ratio, k_frac=k_frac, seed="cuda-fallback", T=T, noise_std=noise_std)
    if not torch.cuda.is_available() and device == "cuda":
        return cs_encode_decode_background_dict_overlap(bg, bs=bs, stride=stride, measure_ratio=measure_ratio, k_frac=k_frac, seed="cuda-fallback", T=T, noise_std=noise_std)
    h, w = bg.shape
    h2 = h - (h - bs) % stride
    w2 = w - (w - bs) % stride
    canvas = np.zeros((h2, w2), dtype=np.float32)
    weight = np.zeros((h2, w2), dtype=np.float32)
    n = bs * bs
    m = int(max(1, round(measure_ratio * n)))
    Phi_np = generate_measurement_matrix(m, n)
    Phi = torch.from_numpy(Phi_np).to(device)
    T_t = torch.from_numpy(T.astype(np.float32)).to(device)
    k = max(1, int(round(k_frac * n)))
    Y_list = []
    for i in range(0, h2 - bs + 1, stride):
        for j in range(0, w2 - bs + 1, stride):
            mb = mask[i : i + bs, j : j + bs]
            roi_ratio = float(np.count_nonzero(mb)) / float(bs * bs * 255) if np.max(mb) > 0 else 0.0
            if roi_ratio < 0.2:
                b = bg[i : i + bs, j : j + bs]
                x = torch.from_numpy(b.flatten().astype(np.float32)).to(device)
                residual = x.clone()
                idxs = []
                A = None
                for _ in range(k):
                    corr = torch.matmul(T_t.T, residual)
                    jmax = int(torch.argmax(torch.abs(corr)).item())
                    if jmax in idxs:
                        break
                    idxs.append(jmax)
                    A_dict = T_t[:, idxs]
                    sol_dict = torch.linalg.lstsq(A_dict, x).solution
                    residual = x - A_dict @ sol_dict
                    if torch.norm(residual).item() < 1e-3:
                        break
                codes = torch.zeros(n, device=device, dtype=torch.float32)
                if len(idxs) > 0:
                    codes[idxs] = sol_dict
                y = Phi @ codes
                if noise_std > 0:
                    y += torch.randn_like(y) * noise_std
                residual_y = y.clone()
                idxs_y = []
                for _ in range(k):
                    corr_y = torch.matmul(Phi.T, residual_y)
                    jmax_y = int(torch.argmax(torch.abs(corr_y)).item())
                    if jmax_y in idxs_y:
                        break
                    idxs_y.append(jmax_y)
                    A_y = Phi[:, idxs_y]
                    sol_y = torch.linalg.lstsq(A_y, y).solution
                    residual_y = y - A_y @ sol_y
                    if torch.norm(residual_y).item() < 1e-3:
                        break
                codes_hat = torch.zeros(n, device=device, dtype=torch.float32)
                if len(idxs_y) > 0:
                    codes_hat[idxs_y] = sol_y
                rec = (T_t @ codes_hat).reshape(bs, bs)
                b_rec = rec.detach().cpu().numpy()
                b_rec = np.clip(b_rec, 0, 255).astype(np.float32)
                canvas[i : i + bs, j : j + bs] += b_rec
                weight[i : i + bs, j : j + bs] += 1.0
                Y_list.append(y.detach().cpu().numpy().astype(np.float32).tobytes())
            else:
                b = bg[i : i + bs, j : j + bs].astype(np.float32)
                canvas[i : i + bs, j : j + bs] += b
                weight[i : i + bs, j : j + bs] += 1.0
    recon = canvas / np.maximum(weight, 1e-6)
    recon = cv2.bilateralFilter(recon.astype(np.uint8), d=5, sigmaColor=25, sigmaSpace=25)
    import zlib
    blob = b"".join(Y_list)
    blob_comp = zlib.compress(blob, 9)
    return recon, blob_comp
