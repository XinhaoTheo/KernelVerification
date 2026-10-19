"""Standalone CUDA sliding-window state update.

The public contract and independent reference are in this file and problem.txt.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _slide(STATE, EVENTS, OUT, L: tl.constexpr, D: tl.constexpr,
           K: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL
    batch = index // (L * D)
    row = (index // D) % L
    col = index % D
    from_state = row < L - K
    old_value = tl.load(STATE + batch * L * D + (row + K) * D + col,
                        mask=valid & from_state, other=0.0)
    new_value = tl.load(EVENTS + batch * K * D + (row - (L - K)) * D + col,
                        mask=valid & ~from_state, other=0.0)
    value = tl.where(from_state, old_value, new_value)
    tl.store(OUT + index, value, mask=valid)


def run(state, new_events):
    """Update state in place, return that same tensor, preserve new_events."""
    assert state.is_cuda and new_events.is_cuda
    assert state.device == new_events.device
    assert state.dtype == new_events.dtype == torch.float32
    assert state.is_contiguous() and new_events.is_contiguous()
    assert state.ndim == new_events.ndim == 3
    batch, length, width = state.shape
    assert new_events.shape[0] == batch and new_events.shape[2] == width
    steps = new_events.shape[1]
    assert 1 <= batch <= 4 and 2 <= length <= 4096 and 1 <= width <= 1024
    assert 1 <= steps <= min(length, 128)
    output = state
    _slide[(triton.cdiv(state.numel(), 1024),)](
        state, new_events, output, length, width, steps, state.numel(),
        BLOCK=1024, num_warps=4,
    )
    return state


def reference(state, new_events):
    """Independent result computed from the pre-call state, without mutation."""
    return torch.cat((state[:, new_events.shape[1]:, :], new_events), dim=1)


def make_inputs(device="cuda", batch=4, length=2048, width=1024,
                steps=2, seed=0):
    """One convenient example; any legal input may be tested."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn((batch, length, width), generator=generator)
    new_events = torch.randn((batch, steps, width), generator=generator)
    return state.to(device), new_events.to(device)
