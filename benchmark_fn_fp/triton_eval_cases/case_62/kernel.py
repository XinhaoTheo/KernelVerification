import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _fit_predict(X, Y, Q, Out, N: tl.constexpr):
    s0 = tl.full((), 0.0, tl.float32)
    s1 = tl.full((), 0.0, tl.float32)
    sy = tl.full((), 0.0, tl.float32)
    s00 = tl.full((), 0.0, tl.float32)
    s01 = tl.full((), 0.0, tl.float32)
    s11 = tl.full((), 0.0, tl.float32)
    s0y = tl.full((), 0.0, tl.float32)
    s1y = tl.full((), 0.0, tl.float32)
    for i in tl.static_range(N):
        x0 = tl.load(X + 2 * i)
        x1 = tl.load(X + 2 * i + 1)
        val = tl.load(Y + i)
        s0 = s0 + x0
        s1 = s1 + x1
        sy = sy + val
        s00 = s00 + x0 * x0
        s01 = s01 + x0 * x1
        s11 = s11 + x1 * x1
        s0y = s0y + x0 * val
        s1y = s1y + x1 * val
    m0 = s0 / N
    m1 = s1 / N
    my = sy / N
    a = (s00 / N - m0 * m0) + 0.0009765625
    b = s01 / N - m0 * m1
    c = (s11 / N - m1 * m1) + 0.0009765625
    u = s0y / N - m0 * my
    v = s1y / N - m1 * my
    det = a * c - b * b
    beta0 = tl.div_rn(c * u - b * v, det)
    beta1 = tl.div_rn(a * v - b * u, det)
    rows = tl.arange(0, 4)
    q0 = tl.load(Q + rows * 2)
    q1 = tl.load(Q + rows * 2 + 1)
    result = (my + (q0 - m0) * beta0) + (q1 - m1) * beta1
    tl.store(Out + rows, result)


def run(x, y, q):
    output = torch.empty((4,), device=x.device, dtype=torch.float32)
    _fit_predict[(1,)](x, y, q, output, N=32, num_warps=1,
                      enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(130555))
    latent = rng.normal(0.0, 0.0625, (32, 2))
    x = (np.asarray([32.0, -24.0]) + latent).astype(np.float32)
    y = (latent @ np.asarray([0.75, -1.25]) +
         rng.normal(0.0, 0.015625, 32)).astype(np.float32)
    q = (np.asarray([32.0, -24.0]) +
         rng.normal(0.0, 0.25, (4, 2))).astype(np.float32)
    return x, y, q


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())


def initial_probe():
    x, y, q = make_inputs()
    actual = run(x, y, q)
    stats = torch.zeros((8,), dtype=torch.float32, device=x.device)
    for i in range(32):
        x0, x1, yi = x[i, 0], x[i, 1], y[i]
        stats = stats + torch.stack((x0, x1, yi, x0*x0, x0*x1,
                                      x1*x1, x0*yi, x1*yi))
    m0, m1, my, second00, second01, second11, cross0, cross1 = stats / 32
    cov00 = second00 - m0*m0 + 1/1024
    cov01 = second01 - m0*m1
    cov11 = second11 - m1*m1 + 1/1024
    rhs0, rhs1 = cross0 - m0*my, cross1 - m1*my
    determinant = cov00*cov11 - cov01*cov01
    b0 = (cov11*rhs0 - cov01*rhs1) / determinant
    b1 = (cov00*rhs1 - cov01*rhs0) / determinant
    expected = (my + (q[:, 0]-m0)*b0) + (q[:, 1]-m1)*b1
    error = torch.linalg.vector_norm(actual.double()-expected.double()) / torch.clamp(
        torch.linalg.vector_norm(expected.double()), min=0.1)
    return {"output": actual.detach().cpu().tolist(),
            "reference": expected.detach().cpu().tolist(),
            "relative_error": float(error.item()),
            "tolerance": 0.05, "passed": bool(error.item() <= 0.05)}
