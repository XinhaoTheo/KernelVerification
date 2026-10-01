import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _rbf_kernel(Query, Anchors, Values, Output, N: tl.constexpr, D: tl.constexpr):
    row = tl.arange(0, N)
    anchor_norm = tl.full((N,), 0.0, tl.float32)
    query_norm = tl.full((), 0.0, tl.float32)
    dot = tl.full((N,), 0.0, tl.float32)
    for k in tl.static_range(0, D):
        anchor = tl.load(Anchors + row * D + k).to(tl.float32)
        query = tl.load(Query + k).to(tl.float32)
        anchor_square = anchor * anchor
        query_square = query * query
        cross = anchor * query
        anchor_norm = anchor_norm + anchor_square
        query_norm = query_norm + query_square
        dot = dot + cross
    norm_sum = anchor_norm + query_norm
    twice_dot = 2.0 * dot
    distance = tl.maximum(norm_sum - twice_dot, 0.0)
    weight = tl.exp(-16.0 * distance)
    value = tl.load(Values + row).to(tl.float32)
    numerator = tl.sum(weight * value, 0)
    denominator = tl.sum(weight, 0)
    output = tl.div_rn(numerator, denominator)
    tl.store(Output, output)


def run(query, anchors, values):
    output = torch.empty((1,), device=query.device, dtype=torch.float32)
    _rbf_kernel[(1,)](query, anchors, values, output, N=16, D=32,
                      num_warps=1, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(119130))
    query = (16.0 + rng.normal(0.0, 0.5, 32)).astype(np.float32)
    anchors = (query.astype(np.float64)[None, :] +
               rng.normal(0.0, 0.015625, (16, 32))).astype(np.float32)
    values = rng.normal(0.0, 1.0, 16).astype(np.float32)
    return query, anchors, values


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
