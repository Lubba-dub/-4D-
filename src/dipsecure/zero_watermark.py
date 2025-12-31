import numpy as np
import pywt
try:
    from reedsolo import RSCodec
except Exception:
    RSCodec = None


def dwt_ll_signature(img: np.ndarray, size: int = 8) -> int:
    img = img.astype(np.float32)
    LL, (LH, HL, HH) = pywt.dwt2(img, "haar")
    h, w = LL.shape
    bh, bw = h // size, w // size
    LL = LL[: bh * size, : bw * size]
    bits = 0
    global_med = np.median(LL)
    idx = 0
    for i in range(size):
        for j in range(size):
            block = LL[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            bits <<= 1
            if block.mean() > global_med:
                bits |= 1
            idx += 1
    return bits


def ecc_encode_repetition(bits: int, size: int = 64, k: int = 3) -> int:
    out = 0
    for i in range(size):
        b = (bits >> (size - 1 - i)) & 1
        for _ in range(k):
            out = (out << 1) | b
    return out


def ecc_decode_repetition(code: int, size: int = 64, k: int = 3) -> int:
    out = 0
    for i in range(size):
        group = 0
        for _ in range(k):
            bitpos = (size * k - 1) - (i * k + _)
            b = (code >> bitpos) & 1
            group = (group << 1) | b
        ones = bin(group).count("1")
        out = (out << 1) | (1 if ones >= (k // 2 + 1) else 0)
    return out


def rs_encode(bits: int, nsym: int = 16) -> bytes:
    if RSCodec is None:
        raise RuntimeError("reedsolo not installed")
    data = bits.to_bytes(8, byteorder="big")
    rsc = RSCodec(nsym)
    code = rsc.encode(data)
    return code


def rs_decode(code: bytes, nsym: int = 16) -> int:
    if RSCodec is None:
        raise RuntimeError("reedsolo not installed")
    rsc = RSCodec(nsym)
    data = rsc.decode(code)[0]
    return int.from_bytes(data, byteorder="big")
