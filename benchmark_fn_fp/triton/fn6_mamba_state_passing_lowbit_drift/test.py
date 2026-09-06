import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import state_passing_lowbit  # buggy: per-chunk state rounding

import triton
import triton.language as tl

@triton.jit
def _state_passing_fwd_kernel_ref(
    states_ptr, out_ptr, final_states_ptr, dA_cs_ptr,
    dim, nchunks,
    stride_states_chunk, stride_states_dim,
    stride_out_chunk, stride_out_dim,
    stride_final_dim,
    stride_dA_cs_chunk,
    BLOCK_SIZE: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_dim

    states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        states = scale * states + new_states  # real mamba recurrence, unmodified
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def state_passing_reference(new_states: torch.Tensor, dA_cs: torch.Tensor) -> torch.Tensor:
    """new_states/dA_cs: (nchunks, dim) / (nchunks,). Returns final_states: (dim,)."""
    nchunks, dim = new_states.shape
    out = torch.empty(nchunks, dim, device=new_states.device, dtype=torch.float32)
    final_states = torch.empty(dim, device=new_states.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(dim)
    _state_passing_fwd_kernel_ref[(1,)](
        new_states, out, final_states, dA_cs,
        dim, nchunks,
        new_states.stride(0), new_states.stride(1),
        out.stride(0), out.stride(1),
        final_states.stride(0),
        dA_cs.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return final_states


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    dim = 8
    decay = -0.001  # log-decay per chunk (scale = exp(decay) ~ 0.999)

    torch.manual_seed(0)
    new_states_short = torch.randn(5, dim, device=device) * 0.01
    dA_cs_short = torch.full((5,), decay, device=device)
    ref_short = state_passing_reference(new_states_short, dA_cs_short)
    cand_short = state_passing_lowbit(new_states_short, dA_cs_short)
    short_pass = torch.allclose(cand_short, ref_short, rtol=1e-2, atol=1e-2)

    new_states_long = torch.randn(2000, dim, device=device) * 0.01
    dA_cs_long = torch.full((2000,), decay, device=device)
    ref_long = state_passing_reference(new_states_long, dA_cs_long)
    cand_long = state_passing_lowbit(new_states_long, dA_cs_long)
    long_pass = torch.allclose(cand_long, ref_long, rtol=1e-2, atol=1e-2)

    print(f"real mamba kernel (GPU), nchunks=5: allclose={short_pass}, diff={(cand_short-ref_short).abs().max().item():.6f}")
    print(f"real mamba kernel (GPU), nchunks=2000: allclose={long_pass}, diff={(cand_long-ref_long).abs().max().item():.6f}")
    print("FN DEMONSTRATED" if short_pass and not long_pass else "tune constants")
    # What a conventional CI test would conclude: 短序列，误差还没积累起来
    print(f"NAIVE_ALLCLOSE_VERDICT: {short_pass}")
    return short_pass and not long_pass


if __name__ == "__main__":
    test_kernel()
