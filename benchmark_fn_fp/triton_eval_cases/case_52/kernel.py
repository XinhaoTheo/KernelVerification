import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _route_kernel(Query, Candidates, Embeddings, Output, N: tl.constexpr,
                  D: tl.constexpr, E: tl.constexpr):
    row = tl.arange(0, N)
    distances = tl.full((N,), 0.0, tl.float32)
    for j in tl.static_range(0, D):
        query = tl.load(Query + j).to(tl.float32)
        candidate = tl.load(Candidates + row * D + j).to(tl.float32)
        query_q = tl.floor(query * 8.0 + 0.5) * 0.125
        candidate_q = tl.floor(candidate * 8.0 + 0.5) * 0.125
        delta = candidate_q - query_q
        distances = distances + delta * delta
    minimum = tl.min(distances, axis=0)
    winner = tl.min(tl.where(distances == minimum, row, 2147483647), axis=0)
    component = tl.arange(0, E)
    result = tl.load(Embeddings + winner * E + component)
    tl.store(Output + component, result)


def run(query, candidates, embeddings):
    output = torch.empty((8,), device=query.device, dtype=torch.float32)
    _route_kernel[(1,)](query, candidates, embeddings, output, N=16, D=32, E=8,
                        num_warps=1, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(840104))
    query = rng.normal(0.0, 0.3, 32).astype(np.float32)
    offsets = rng.normal(size=(16, 32))
    offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)
    radii = 1.0 + rng.uniform(-0.002, 0.002, 16)
    candidates = (query.astype(np.float64)[None, :] + offsets * radii[:, None]).astype(np.float32)
    embeddings = rng.normal(size=(16, 8)).astype(np.float32)
    return query, candidates, embeddings


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
