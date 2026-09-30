import torch
import triton
import triton.language as tl


@triton.jit
def _route_kernel(Logits, Idx, stride, E: tl.constexpr, BLOCK: tl.constexpr):
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < E
    x = tl.load(Logits + tok * stride + cols, mask=valid, other=-float("inf"))

    best = -float("inf")
    best_i = 0
    for e in range(E):
        v = tl.sum(tl.where(cols == e, x, 0.0), axis=0)
        take = v >= best
        best = tl.where(take, v, best)
        best_i = tl.where(take, e, best_i)
    tl.store(Idx + tok, best_i)


def route_top1(logits):
    # Pick one expert per token from the router logits.
    n_tokens, n_experts = logits.shape
    idx = torch.empty(n_tokens, device=logits.device, dtype=torch.int32)
    _route_kernel[(n_tokens,)](logits, idx, logits.stride(0), E=n_experts,
                               BLOCK=triton.next_power_of_2(n_experts))
    return idx
