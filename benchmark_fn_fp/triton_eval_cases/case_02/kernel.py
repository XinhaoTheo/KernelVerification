import torch
import triton
import triton.language as tl


@triton.jit
def _state_passing_fwd_kernel(
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
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def state_passing(new_states: torch.Tensor, dA_cs: torch.Tensor) -> torch.Tensor:
    nchunks, dim = new_states.shape
    out = torch.empty(nchunks, dim, device=new_states.device, dtype=torch.float32)
    final_states = torch.empty(dim, device=new_states.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(dim)
    _state_passing_fwd_kernel[(1,)](
        new_states, out, final_states, dA_cs,
        dim, nchunks,
        new_states.stride(0), new_states.stride(1),
        out.stride(0), out.stride(1),
        final_states.stride(0),
        dA_cs.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return final_states
