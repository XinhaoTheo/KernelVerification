import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    mean = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        mean += tl.sum(a, axis=0)
    mean = mean / N

    var = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        a = tl.where(cols < N, a - mean, 0.0)
        var += tl.sum(a * a, axis=0)
    var = var / N
    rstd = 1.0 / (tl.sqrt(var) + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + cols, (a - mean) * rstd * w + b, mask=mask)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
               eps: float = 1e-5) -> torch.Tensor:
    """Row-wise layer normalization with affine parameters."""
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    _layer_norm_kernel[(n_rows,)](x, y, weight, bias, mean, rstd,
                                  x.stride(0), n_cols, eps, BLOCK=256)
    return y
