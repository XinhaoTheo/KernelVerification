import torch
import triton
import triton.language as tl

_TIE_EPS = 1e-3


@triton.jit
def _update_min_larger_stats(data, above_mask, min_larger, num_min_larger, sentinel, EPS: tl.constexpr):
    tile_min = tl.min(tl.where(above_mask, data, sentinel))
    tile_eq = above_mask & (tl.abs(data - tile_min) < EPS)
    tile_cnt = tl.sum(tile_eq)
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < EPS
    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def _count_tied_at_boundary_kernel(scores_ptr, count_ptr, N: tl.constexpr, pivot, EPS: tl.constexpr):
    offs = tl.arange(0, N)
    data = tl.load(scores_ptr + offs)
    above_mask = data > pivot
    min_larger = tl.full((), float("inf"), tl.float32)
    num_min_larger = tl.zeros((), tl.int32)
    min_larger, num_min_larger = _update_min_larger_stats(data, above_mask, min_larger, num_min_larger, float("inf"), EPS)
    tl.store(count_ptr, num_min_larger)


def count_tied_at_boundary(scores: torch.Tensor, pivot: float) -> int:
    """Real vLLM primitive: how many values above `pivot` are tied with the
    smallest of them (within EPS). A real multi-pass top-k kernel uses this
    count to know how many boundary candidates remain to fill the last slots."""
    N = scores.shape[0]
    count = torch.empty(1, dtype=torch.int32, device=scores.device)
    _count_tied_at_boundary_kernel[(1,)](scores, count, N=N, pivot=pivot, EPS=_TIE_EPS)
    return int(count.item())
