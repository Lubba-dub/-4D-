import numpy as np


def rk4_step(f, state, h, params):
    k1 = f(state, params)
    k2 = f(state + 0.5 * h * k1, params)
    k3 = f(state + 0.5 * h * k2, params)
    k4 = f(state + h * k3, params)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz4d_deriv(state, params):
    # Simple 4D Lorenz-like system
    # dx/dt = a*(y - x)
    # dy/dt = x*(b - z) - y
    # dz/dt = x*y - c*z + w
    # dw/dt = -d*w + e*x
    x, y, z, w = state
    a, b, c, d, e = params
    dx = a * (y - x)
    dy = x * (b - z) - y
    dz = x * y - c * z + w
    dw = -d * w + e * x
    return np.array([dx, dy, dz, dw], dtype=np.float64)


def generate_stream(length, x0=0.1, y0=0.2, z0=0.3, w0=0.4, params=(10.0, 28.0, 8.0 / 3.0, 1.0, 2.5), h=0.01, discard=1000):
    state = np.array([x0, y0, z0, w0], dtype=np.float64)
    # Burn-in
    for _ in range(discard):
        state = rk4_step(lorenz4d_deriv, state, h, params)
    seq = np.zeros((length, 4), dtype=np.float64)
    for i in range(length):
        state = rk4_step(lorenz4d_deriv, state, h, params)
        seq[i] = state
    # Whitening & scaling to bytes
    flat = seq.reshape(-1)
    flat = (flat - np.mean(flat)) / (np.std(flat) + 1e-8)
    # Map to 0-255 using erf for quasi-uniform
    from math import erf, sqrt

    def to_uint8(v):
        u = 0.5 * (1.0 + erf(v / sqrt(2.0)))  # approx CDF of N(0,1)
        return int(max(0, min(255, round(u * 255))))

    out = np.fromiter((to_uint8(v) for v in flat), dtype=np.uint8, count=len(flat))
    return out


def generate_measurement_matrix(m, n, params=(10.0, 28.0, 8.0 / 3.0, 1.0, 2.5), x0=0.11, y0=0.22, z0=0.33, w0=0.44, h=0.005):
    # Use chaotic stream to fill matrix, then normalize columns
    length = m * n * 4
    stream = generate_stream(length, x0=x0, y0=y0, z0=z0, w0=w0, params=params, h=h)
    mat = stream[: m * n].astype(np.float32)
    mat = mat.reshape(m, n)
    # Normalize columns for better OMP behavior
    mat = mat - np.mean(mat, axis=0, keepdims=True)
    std = np.std(mat, axis=0, keepdims=True) + 1e-6
    mat = mat / std
    return mat.astype(np.float32)

