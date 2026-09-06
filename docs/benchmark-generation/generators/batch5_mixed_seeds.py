"""Batch 5: six cases across FN4, FN6, FN7, FP2, FP3, FP6.

Two recipes have already been falsified by measurement, and both failures are
recorded here so they are not retried:

  - a small clean kernel with a divisibility bug (batch 4): the single-call
    baseline read `n_cols // BLOCK` straight off the page, 4 of 4, conf 0.98.
  - a 354-line verbatim Liger kernel with a one-character tie-break change
    (fn12): also read straight off the page, conf 0.93.

The remaining hypothesis, taken from the one case that does defeat it
(fn3_gptq), is that correctness has to depend on a relationship between INPUT
TENSORS that the source cannot show -- an indirection table's length, a page
table's layout, a scale array's coverage. There the reader has to assume, and
only a run settles it. fn13 below tests that directly; the other five fill out
seeds that are currently single-sample.

Usage (from repo root):
    python docs/benchmark-generation/generators/batch5_mixed_seeds.py
"""
from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO / "benchmark_fn_fp" / "triton"

# ==========================================================================
# fn13 -- FN4: paged KV gather through a block table
# ==========================================================================
PAGED_KERNEL = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_gather_kernel(KV, BlockTable, Out, max_blocks, page_size,
                         stride_kv_page, stride_out_seq,
                         HEAD_DIM: tl.constexpr):
    seq = tl.program_id(0)
    slot = tl.program_id(1)
    logical_block = slot // page_size
    within = slot % page_size

    physical = seq * max_blocks + logical_block
    offs = tl.arange(0, HEAD_DIM)
    src = KV + physical * stride_kv_page + within * HEAD_DIM + offs
    dst = Out + seq * stride_out_seq + slot * HEAD_DIM + offs
    tl.store(dst, tl.load(src))


def paged_gather(kv_cache: torch.Tensor, block_table: torch.Tensor,
                 seq_len: int) -> torch.Tensor:
    """Gather `seq_len` cached tokens for each sequence through its block table."""
    num_pages, page_size, head_dim = kv_cache.shape
    num_seqs, max_blocks = block_table.shape
    out = torch.empty((num_seqs, seq_len, head_dim), device=kv_cache.device,
                      dtype=kv_cache.dtype)
    _paged_gather_kernel[(num_seqs, seq_len)](
        kv_cache, block_table, out, max_blocks, page_size,
        kv_cache.stride(0), out.stride(0), HEAD_DIM=head_dim)
    return out
'''

PAGED_PROBLEM = """Operation: gather cached key/value tokens for a batch of sequences out of a
paged KV cache, the block-table indirection used by paged-attention inference
engines.

    kv_cache     : [num_pages, page_size, head_dim]
    block_table  : [num_seqs, max_blocks], int32
                   block_table[s, b] is the PHYSICAL page holding logical block b
                   of sequence s
    result[s, t] = kv_cache[ block_table[s, t // page_size], t % page_size ]

Pages are allocated on demand, so the physical pages of one sequence are not
contiguous and are generally interleaved with those of other sequences.

Input domain: `seq_len` may be any length that fits within `max_blocks` pages,
and the block table may map any sequence to any physical page.

Does paged_gather() satisfy this contract?
"""

PAGED_TEST = '''
import torch
from kernel import paged_gather


def reference(kv_cache, block_table, seq_len):
    page_size = kv_cache.shape[1]
    num_seqs = block_table.shape[0]
    out = torch.empty((num_seqs, seq_len, kv_cache.shape[2]), device=kv_cache.device,
                      dtype=kv_cache.dtype)
    for s in range(num_seqs):
        for t in range(seq_len):
            out[s, t] = kv_cache[block_table[s, t // page_size].item(), t % page_size]
    return out


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    num_pages, page_size, head_dim = 32, 8, 16
    num_seqs, max_blocks, seq_len = 4, 4, 32

    # --- test data ------------------------------------------------------------
    kv = torch.randn(num_pages, page_size, head_dim, device=device)
    kv += torch.arange(num_pages, device=device).view(-1, 1, 1) * 10.0

    # Lazy test data: pages happen to be handed out contiguously in order.
    identity = torch.arange(num_seqs * max_blocks, device=device,
                            dtype=torch.int32).reshape(num_seqs, max_blocks)
    cand_h = paged_gather(kv, identity, seq_len)
    ref_h = reference(kv, identity, seq_len)
    homogeneous_pass = torch.allclose(cand_h, ref_h, rtol=1e-2, atol=1e-2)

    # Contract-permitted table: pages of different sequences are interleaved,
    # which is what a real allocator produces once requests come and go.
    perm = torch.randperm(num_pages, device=device)[:num_seqs * max_blocks]
    shuffled = perm.to(torch.int32).reshape(num_seqs, max_blocks)
    cand_d = paged_gather(kv, shuffled, seq_len)
    ref_d = reference(kv, shuffled, seq_len)
    distinct_pass = torch.allclose(cand_d, ref_d, rtol=1e-2, atol=1e-2)
    diff = (cand_d - ref_d).abs().max().item()

    print(f"pages handed out in order: allclose={homogeneous_pass}")
    print(f"pages interleaved by the allocator: allclose={distinct_pass}, max abs diff={diff:.4f}")
    # What a conventional CI test would conclude: it fills the cache with one
    # repeated pattern, so reading the wrong page returns the same values.
    print(f"NAIVE_ALLCLOSE_VERDICT: {homogeneous_pass}")
    return homogeneous_pass and not distinct_pass


if __name__ == "__main__":
    test_kernel()
'''

# ==========================================================================
# fn14 -- FN7: LayerNorm, eps added outside the square root
# ==========================================================================
LN_KERNEL = '''
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
'''

LN_PROBLEM = """Operation: layer normalization forward
(Dao-AILab/flash-attention, flash_attn/ops/triton/layer_norm.py; the same
formula is used by Liger-Kernel and by every transformer implementation).

For each row x:

    mean = sum(x) / N
    var  = sum((x - mean)^2) / N
    rstd = 1 / sqrt(var + eps)
    y    = (x - mean) * rstd * weight + bias

The epsilon is the standard stabilizer: it must keep the normalizer finite as
the row variance approaches zero, so that a constant row does not blow the
output up.

Does layer_norm() satisfy this contract?
"""

LN_TEST = '''
import torch
from kernel import layer_norm


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_cols, eps = 512, 1e-5
    w = torch.ones(n_cols, device=device)
    b = torch.zeros(n_cols, device=device)

    x_normal = torch.randn(8, n_cols, device=device)
    cand_n = layer_norm(x_normal, w, b, eps)
    ref_n = torch.nn.functional.layer_norm(x_normal, (n_cols,), w, b, eps)
    normal_pass = torch.allclose(cand_n, ref_n, rtol=1e-2, atol=1e-2)

    # A row whose variance is far below eps, but whose deviations are still
    # representable, so the normalized output is not identically zero.
    x_flat = torch.full((1, n_cols), 1e-4, device=device)
    x_flat[:, 1::2] = -1e-4
    cand_f = layer_norm(x_flat, w, b, eps)
    ref_f = torch.nn.functional.layer_norm(x_flat, (n_cols,), w, b, eps)
    flat_pass = torch.allclose(cand_f, ref_f, rtol=1e-2, atol=1e-2)

    print(f"normal rows: allclose={normal_pass}")
    print(f"near-constant row: allclose={flat_pass}, ref max={ref_f.abs().max().item():.3e}, cand max={cand_f.abs().max().item():.3e}")
    # What a conventional CI test would conclude: Gaussian rows never have
    # near-zero variance.
    print(f"NAIVE_ALLCLOSE_VERDICT: {normal_pass}")
    return normal_pass and not flat_pass


if __name__ == "__main__":
    test_kernel()
'''

# ==========================================================================
# fn15 -- FN6: repeated INT8 requantization drifts over many steps
# ==========================================================================
REQUANT_KERNEL = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _requant_step_kernel(X, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    absmax = tl.max(tl.abs(x), axis=0)
    scale = tl.where(absmax == 0.0, 1.0, absmax / 127.0)
    q = tl.floor(x / scale).to(tl.int8)
    tl.store(OUT + offs, q.to(tl.float32) * scale, mask=mask)


def requantize(x: torch.Tensor) -> torch.Tensor:
    """One symmetric INT8 quantize-dequantize round trip over a 1-D tensor."""
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    _requant_step_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
    return out
'''

REQUANT_PROBLEM = """Operation: one symmetric INT8 quantize-dequantize round trip, the storage step
used by low-precision optimizer state and KV-cache compression.

    scale = max(|x|) / 127        per block
    q     = round(x / scale)      to the nearest representable level
    y     = q * scale

The rounding must be to nearest, so the round trip is unbiased: repeated
application must not push the tensor systematically toward zero or away from it.
This matters because the operation is applied once per optimizer step, so any
per-step bias compounds over a training run.

Does requantize() satisfy this contract?
"""

REQUANT_TEST = '''
import torch
from kernel import requantize


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    x0 = torch.randn(4096, device=device)

    # Conventional test: one round trip, compared with a tolerance.
    once = requantize(x0.clone())
    # An INT8 round trip is only accurate to about absmax/127 per element, so a
    # test written for this kernel uses a tolerance matched to the format.
    level = x0.abs().max().item() / 127.0
    short_pass = torch.allclose(once, x0, rtol=0.0, atol=2 * level)
    one_step_bias = (once.abs().sum() / x0.abs().sum()).item()

    # Long-horizon behaviour: one optimizer step is "add the gradient, then store
    # the parameter back in INT8". A per-step rounding bias compounds here.
    grad = torch.full_like(x0, 1e-3)
    x = x0.clone()
    x_ref = x0.clone()
    for _ in range(2000):
        x = requantize(x + grad)
        x_ref = x_ref + grad
    long_pass = torch.allclose(x, x_ref, rtol=0.0, atol=2 * level)
    tracked = ((x - x0).mean() / (x_ref - x0).mean()).item()

    print(f"1 round trip: allclose={short_pass}, magnitude ratio={one_step_bias:.6f}")
    print(f"2000 optimizer steps: allclose={long_pass}, fraction of the update actually tracked={tracked:.4f}")
    # What a conventional CI test would conclude: a kernel unit test runs the
    # operation once.
    print(f"NAIVE_ALLCLOSE_VERDICT: {short_pass}")
    return short_pass and not long_pass


if __name__ == "__main__":
    test_kernel()
'''

# ==========================================================================
# fp7 -- FP2: packed QKV layout convention
# ==========================================================================
QKV_KERNEL = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _split_qkv_kernel(QKV, Q, K, V, n_tokens, stride_t,
                      HEADS: tl.constexpr, DIM: tl.constexpr):
    tok = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, DIM)
    base = QKV + tok * stride_t + h * 3 * DIM
    q = tl.load(base + offs)
    k = tl.load(base + DIM + offs)
    v = tl.load(base + 2 * DIM + offs)
    out_off = tok * HEADS * DIM + h * DIM + offs
    tl.store(Q + out_off, q)
    tl.store(K + out_off, k)
    tl.store(V + out_off, v)


def split_qkv(qkv: torch.Tensor, num_heads: int):
    """Split a packed QKV activation into separate Q, K and V tensors."""
    n_tokens, total = qkv.shape
    dim = total // (3 * num_heads)
    q = torch.empty((n_tokens, num_heads * dim), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    _split_qkv_kernel[(n_tokens, num_heads)](qkv, q, k, v, n_tokens, qkv.stride(0),
                                             HEADS=num_heads, DIM=dim)
    return q, k, v
'''

QKV_PROBLEM = """Operation: split a packed QKV activation into separate Q, K and V tensors, the
projection-output unpacking used by fused attention kernels.

A packed QKV tensor stores, for every token, the query, key and value vectors of
every head. Two packings are in common use and both appear in production
transformer code:

    head-major   : [head0_q, head0_k, head0_v, head1_q, head1_k, head1_v, ...]
    tensor-major : [head0_q, head1_q, ..., head0_k, head1_k, ..., head0_v, ...]

The projection weights are laid out to match whichever packing the model was
exported with, so the packing is a property of the checkpoint, not of the
attention math. A splitter is correct if it is self-consistent: attention scores
computed from its Q and K must equal the scores computed from the same tokens
under the reference packing, once both sides use their own convention.

Does split_qkv() satisfy this contract?
"""

QKV_TEST = '''
import torch
from kernel import split_qkv


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_tokens, heads, dim = 32, 4, 16

    qkv = torch.randn(n_tokens, 3 * heads * dim, device=device)
    q_c, k_c, v_c = split_qkv(qkv, heads)

    # Reference splitter using the OTHER packing convention.
    ref = qkv.view(n_tokens, 3, heads, dim)
    q_r = ref[:, 0].reshape(n_tokens, heads * dim)
    k_r = ref[:, 1].reshape(n_tokens, heads * dim)
    v_r = ref[:, 2].reshape(n_tokens, heads * dim)

    raw_pass = torch.allclose(q_c, q_r, rtol=1e-2, atol=1e-2)

    # Downstream: attention scores are a permutation-invariant contraction over
    # the head dimension, so a self-consistent splitter reproduces them.
    def scores(q, k):
        q = q.view(n_tokens, heads, dim)
        k = k.view(n_tokens, heads, dim)
        return torch.einsum("qhd,khd->hqk", q, k)

    s_c = scores(q_c, k_c)
    s_r = scores(q_r, k_r)
    downstream_pass = torch.allclose(s_c.sort(dim=0).values, s_r.sort(dim=0).values,
                                     rtol=1e-2, atol=1e-2)

    print(f"raw Q tensor allclose against the other convention = {raw_pass} (expected False)")
    print(f"downstream attention scores agree up to head order = {downstream_pass} (expected True)")
    # What a conventional CI test would conclude: it compares the raw tensors
    # against one reference convention.
    print(f"NAIVE_ALLCLOSE_VERDICT: {raw_pass}")
    return (not raw_pass) and downstream_pass


if __name__ == "__main__":
    test_kernel()
'''

# ==========================================================================
# fp8 -- FP3: split-K accumulation order
# ==========================================================================
SPLITK_KERNEL = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _splitk_dot_kernel(A, B, Out, K, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    part = tl.program_id(0)
    chunk = tl.cdiv(K, SPLIT)
    start = part * chunk
    acc = 0.0
    for off in range(0, chunk, BLOCK):
        idx = start + off + tl.arange(0, BLOCK)
        mask = idx < tl.minimum(start + chunk, K)
        a = tl.load(A + idx, mask=mask, other=0.0)
        b = tl.load(B + idx, mask=mask, other=0.0)
        acc += tl.sum(a * b, axis=0)
    tl.atomic_add(Out, acc)


def splitk_dot(a: torch.Tensor, b: torch.Tensor, split: int = 512) -> torch.Tensor:
    """Inner product of two float32 vectors, accumulated across `split` partitions."""
    out = torch.zeros(1, device=a.device, dtype=torch.float32)
    _splitk_dot_kernel[(split,)](a, b, out, a.numel(), SPLIT=split, BLOCK=256)
    return out
'''

SPLITK_PROBLEM = """Operation: inner product of two float32 vectors, computed with a split-K
partitioning and combined with atomic accumulation, the reduction strategy used
by split-K GEMM and all-reduce kernels.

    out = sum over k of a[k] * b[k]

The partitioning is a performance choice. Floating-point addition is not
associative, so partitioning and combining in a different order changes the
last bits of the result; the contract requires the mathematical inner product,
not bitwise agreement with any particular summation order.

Does splitk_dot() satisfy this contract?
"""

SPLITK_TEST = '''
import torch
from kernel import splitk_dot


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n = 1 << 16
    a = torch.randn(n, device=device)
    b = torch.randn(n, device=device)

    cand = splitk_dot(a, b).item()
    ref_seq = torch.dot(a, b).item()             # one sequential order
    ref_exact = torch.dot(a.double(), b.double()).item()

    # A CI check that requires reproducibility stores a golden value and demands
    # the kernel reproduce it exactly.
    runs = [splitk_dot(a, b).item() for _ in range(5)]
    bitwise_reproducible = len(set(runs)) == 1

    # What the contract actually asks for: the inner product, to within the
    # rounding the summation order can account for.
    budget = 1e-6 * (a.abs() * b.abs()).sum().item()
    all_runs_accurate = all(abs(r - ref_exact) <= budget for r in runs)
    spread = max(runs) - min(runs)

    print(f"5 runs bitwise identical = {bitwise_reproducible} (expected False), spread={spread:.3e}")
    print(f"every run within the reordering budget of the float64 value = {all_runs_accurate} "
          f"(expected True), budget={budget:.3e}, ref={ref_exact:.6f}")
    print(f"one sequential order for comparison: {ref_seq:.6f}")
    # What a conventional CI test would conclude: it demands a reproducible value.
    print(f"NAIVE_ALLCLOSE_VERDICT: {bitwise_reproducible}")
    return (not bitwise_reproducible) and all_runs_accurate


if __name__ == "__main__":
    test_kernel()
'''

# ==========================================================================
# fp9 -- FP6: cosine similarity where both vectors are near zero
# ==========================================================================
COSINE_KERNEL = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _cosine_kernel(A, B, Out, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    a_row = A + row * stride
    b_row = B + row * stride
    dot = 0.0
    na = 0.0
    nb = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        a = tl.load(a_row + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_row + cols, mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(a * b, axis=0)
        na += tl.sum(a * a, axis=0)
        nb += tl.sum(b * b, axis=0)
    denom = tl.sqrt(na) * tl.sqrt(nb)
    tl.store(Out + row, dot / tl.maximum(denom, eps))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    """Row-wise cosine similarity of two 2-D float32 tensors."""
    n_rows, n_cols = a.shape
    out = torch.empty(n_rows, device=a.device, dtype=torch.float32)
    _cosine_kernel[(n_rows,)](a, b, out, a.stride(0), n_cols, eps, BLOCK=256)
    return out
'''

COSINE_PROBLEM = """Operation: row-wise cosine similarity of two 2-D float32 tensors.

    cos(a, b) = dot(a, b) / (||a|| * ||b||)

with the denominator floored at `eps` so a zero-norm row does not divide by
zero. The result is a bounded quantity in [-1, 1]; what matters downstream is
the absolute agreement of that quantity, since it is used directly as a
similarity score.

Does cosine_similarity() satisfy this contract?
"""

COSINE_TEST = '''
import torch
from kernel import cosine_similarity


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_cols = 512

    # Rows whose vectors are near-orthogonal, so the true cosine is ~0.
    a = torch.randn(64, n_cols, device=device)
    r = torch.randn_like(a)
    # Project r onto the orthogonal complement of a: the true cosine is ~0.
    b = r - ((r * a).sum(1, keepdim=True) / (a * a).sum(1, keepdim=True)) * a
    cand = cosine_similarity(a, b)
    ref = torch.nn.functional.cosine_similarity(a, b, dim=1)

    abs_diff = (cand - ref).abs().max().item()
    ref_mag = ref.abs().max().item()
    rel = ((cand - ref).abs() / ref.abs().clamp_min(1e-30)).max().item()

    relative_only_pass = rel <= 1e-2
    combined_pass = torch.allclose(cand, ref, rtol=1e-2, atol=1e-2)

    print(f"true cosine magnitude ~ {ref_mag:.3e}, max abs diff = {abs_diff:.3e}")
    print(f"relative-error-only check = {relative_only_pass} (expected False), max rel = {rel:.3f}")
    print(f"torch.allclose (atol+rtol combined) = {combined_pass} (expected True)")
    # What a conventional CI test would conclude: it uses a relative-error
    # metric, which is ill-conditioned when the true value is ~0.
    print(f"NAIVE_ALLCLOSE_VERDICT: {relative_only_pass}")
    return (not relative_only_pass) and combined_pass


if __name__ == "__main__":
    test_kernel()
'''

CASES = {
    "fn13_paged_kv_block_table_homogeneous": dict(
        kernel=PAGED_KERNEL, problem=PAGED_PROBLEM, test=PAGED_TEST,
        group="FN", seed_class="FN4", failure_mode="false_negative",
        kernel_family="paged KV-cache gather through a block table",
        reference="vllm-project/vllm, paged-attention block-table indirection (vllm/attention/ops/); same structure in SGLang RadixAttention",
        mechanism="The kernel never reads the block table: it computes the physical page as seq * max_blocks + logical_block, i.e. it assumes pages were handed out contiguously in sequence order. Whether that assumption holds is a property of the block_table INPUT, not of the source, so nothing in the kernel text distinguishes a correct gather from a wrong one. A test that builds the table as arange() -- the natural lazy choice -- makes the assumption true and the defect invisible.",
        naive="PASS",
    ),
    "fn14_layer_norm_eps_outside_sqrt": dict(
        kernel=LN_KERNEL, problem=LN_PROBLEM, test=LN_TEST,
        group="FN", seed_class="FN7", failure_mode="false_negative",
        kernel_family="layer normalization forward",
        reference="Dao-AILab/flash-attention, flash_attn/ops/triton/layer_norm.py; identical formula in Liger-Kernel",
        mechanism="rstd is computed as 1/(sqrt(var) + eps) instead of 1/sqrt(var + eps). At ordinary variances the two agree to within float noise; only a row whose variance approaches zero separates them by orders of magnitude.",
        naive="PASS",
    ),
    "fn15_int8_requant_round_toward_zero_drift": dict(
        kernel=REQUANT_KERNEL, problem=REQUANT_PROBLEM, test=REQUANT_TEST,
        group="FN", seed_class="FN6", failure_mode="false_negative",
        kernel_family="symmetric INT8 quantize-dequantize round trip",
        reference="sgl-project/sglang int8_kernel.py and low-precision optimizer-state storage in bitsandbytes",
        mechanism="Rounding uses tl.floor instead of round-to-nearest, so every store is biased down by up to one level. A single round trip stays inside the tolerance; used as the storage step of an optimizer it truncates each small update, and after many steps the parameter has tracked only part of the accumulated gradient.",
        naive="PASS",
    ),
    "fp8_splitk_atomic_accumulation_order": dict(
        kernel=SPLITK_KERNEL, problem=SPLITK_PROBLEM, test=SPLITK_TEST,
        group="FP", seed_class="FP3", failure_mode="false_positive",
        kernel_family="split-K inner product with atomic accumulation",
        reference="split-K GEMM in triton-lang/triton tutorials and vLLM; atomic accumulation in all-reduce kernels",
        mechanism="Partitioning the sum and combining the partials with float atomics makes the addition order depend on how the partitions happen to be scheduled, so the result differs in its low bits between runs. A CI check that pins a golden value rejects the kernel; every run is nonetheless within the rounding the reordering can account for.",
        naive="FAIL",
    ),
    "fp9_cosine_similarity_near_zero": dict(
        kernel=COSINE_KERNEL, problem=COSINE_PROBLEM, test=COSINE_TEST,
        group="FP", seed_class="FP6", failure_mode="false_positive",
        kernel_family="row-wise cosine similarity",
        reference="cosine similarity as used in embedding and retrieval kernels; the metric instability is the one torch.allclose's atol exists for",
        mechanism="For near-orthogonal rows the true cosine is ~0, so a negligible absolute difference becomes an enormous relative error. A relative-error-only comparison rejects a correct kernel; a scale-aware comparison accepts it.",
        naive="FAIL",
    ),
}


def main() -> None:
    for name, spec in CASES.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.py").write_text(f'"""Triton kernel under test: {name}."""' + spec["kernel"])
        (d / "problem.txt").write_text(spec["problem"])
        (d / "test.py").write_text(spec["test"].lstrip("\n"))
        (d / "meta.json").write_text(json.dumps({
            "name": name,
            "benchmark_version": "fn_fp_triton_v1",
            "group": spec["group"],
            "seed_class": spec["seed_class"],
            "failure_mode": spec["failure_mode"],
            "kernel_family": spec["kernel_family"],
            "reference": spec["reference"],
            "mechanism": spec["mechanism"],
            "default_tolerance": {"rtol": 0.01, "atol": 0.01},
            "expected": {
                "ground_truth": spec["mechanism"].split(".")[0],
                "naive_allclose_verdict": spec["naive"],
                "correct_verdict": "BUGGY" if spec["group"] == "FN" else "CORRECT",
            },
            "status": "seed_v2",
            "source": "real_triton_kernel_adapted",
            "requires_gpu": True,
            "passed": None,
        }, indent=2) + "\n")
        print("wrote", name)


if __name__ == "__main__":
    main()
