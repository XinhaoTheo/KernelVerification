"""Batch 4: FN3 (irregular boundaries) on four more real Triton kernel families.

FN3 is the only seed the measured runs showed a difference on: the single-call
baseline got fn3_gptq wrong on 4 of 4 attempts, each time reasoning "this matches
the upstream implementation", while the debate system ran a probe at the
irregular shape and rejected it. These four reuse that shape on different real
kernels, so the finding rests on more than one sample.

Each case: a real kernel family, one divisibility assumption in its indexing, and
a problem.txt that puts an irregular shape explicitly inside the required input
domain. A conventional test uses a round shape and passes.

Usage (from repo root):
    python docs/benchmark-generation/generators/batch4_fn3_boundary.py
"""
from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO / "benchmark_fn_fp" / "triton"

# --------------------------------------------------------------------------
# FN3-a: fused softmax, column loop bound uses floor division
# --------------------------------------------------------------------------
SOFTMAX_KERNEL = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(OUT, IN, stride_om, stride_im, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    in_row = IN + row * stride_im
    out_row = OUT + row * stride_om
    n_full = n_cols // BLOCK

    row_max = -float("inf")
    for b in range(tl.cdiv(n_cols, BLOCK)):
        cols = b * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_row + cols, mask=cols < n_cols, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    denom = 0.0
    for b in range(n_full):
        cols = b * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_row + cols, mask=cols < n_cols, other=-float("inf"))
        denom += tl.sum(tl.exp(x - row_max), axis=0)

    for b in range(tl.cdiv(n_cols, BLOCK)):
        cols = b * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_row + cols, mask=cols < n_cols, other=-float("inf"))
        tl.store(out_row + cols, tl.exp(x - row_max) / denom, mask=cols < n_cols)


def softmax(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Row-wise softmax of a 2-D float32 tensor."""
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](out, x, out.stride(0), x.stride(0), n_cols, BLOCK=block_size)
    return out
'''

SOFTMAX_PROBLEM = """Operation: row-wise softmax of a 2-D float32 tensor, computed in blocks of
`block_size` columns (the standard three-pass fused-softmax structure used by
Triton-based normalization kernels).

For each row x:

    y[j] = exp(x[j] - max(x)) / sum_k exp(x[k] - max(x))

so every row of the result must be a probability distribution: all entries
non-negative and summing to 1.

Input domain: `n_cols` may be any positive integer. It is NOT required to be a
multiple of `block_size`; when it is not, the trailing partial block is still
part of the row and must be normalized with the rest of it.

Does softmax() satisfy this contract?
"""

SOFTMAX_TEST = '''
import torch
from kernel import softmax


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    block = 128

    # Conventional test shape: n_cols is a multiple of the block size.
    x_round = torch.randn(8, 1024, device=device)
    cand_round = softmax(x_round, block)
    ref_round = torch.softmax(x_round, dim=1)
    divisible_pass = torch.allclose(cand_round, ref_round, rtol=1e-2, atol=1e-2)

    # Contract-permitted irregular shape: a trailing partial block exists.
    x_tail = torch.randn(8, 1000, device=device)
    cand_tail = softmax(x_tail, block)
    ref_tail = torch.softmax(x_tail, dim=1)
    irregular_pass = torch.allclose(cand_tail, ref_tail, rtol=1e-2, atol=1e-2)

    # A loose tolerance cannot expose this one even at the irregular shape:
    # softmax entries are ~1/n_cols, so atol=1e-2 swallows the whole error.
    # The contract's own invariant does expose it.
    row_sums = cand_tail.sum(dim=1)
    invariant_holds = bool(torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-3, atol=1e-3))

    print(f"n_cols=1024 (multiple of block): allclose={divisible_pass}")
    print(f"n_cols=1000 (partial tail): allclose={irregular_pass} (loose tolerance still passes), "
          f"row sums {row_sums.min().item():.4f}..{row_sums.max().item():.4f}, "
          f"sums-to-one invariant holds={invariant_holds}")
    # What a conventional CI test would conclude: a round n_cols never creates the
    # trailing block, and even at n_cols=1000 the tolerance comparison passes.
    print(f"NAIVE_ALLCLOSE_VERDICT: {divisible_pass}")
    return divisible_pass and irregular_pass and not invariant_holds


if __name__ == "__main__":
    test_kernel()
'''

# --------------------------------------------------------------------------
# FN3-b: split-K style matmul, K loop unmasked on the tail
# --------------------------------------------------------------------------
MATMUL_KERNEL = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B for float32 operands."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _matmul_kernel[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1),
                         BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return c
'''

MATMUL_PROBLEM = """Operation: dense matrix multiplication C = A @ B for float32 operands, blocked
over M, N and K (the standard tiled-GEMM structure used by Triton matmul
kernels).

    C[m, n] = sum over k of A[m, k] * B[k, n]

Input domain: M, N and K may each be any positive integer. In particular K is
NOT required to be a multiple of the kernel's K block size; when it is not, the
final K tile is partial and only the first (K mod BLOCK_K) of its entries take
part in the sum.

Does matmul() satisfy this contract?
"""

MATMUL_TEST = '''
import torch
from kernel import matmul


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"

    # Conventional test shape: K is a multiple of the K block size (32).
    a_r = torch.randn(128, 256, device=device)
    b_r = torch.randn(256, 128, device=device)
    cand_r = matmul(a_r, b_r)
    ref_r = a_r @ b_r
    divisible_pass = torch.allclose(cand_r, ref_r, rtol=1e-2, atol=1e-2)

    # Contract-permitted irregular K: the last K tile is partial.
    a_i = torch.randn(128, 250, device=device)
    b_i = torch.randn(250, 128, device=device)
    cand_i = matmul(a_i, b_i)
    ref_i = a_i @ b_i
    irregular_pass = torch.allclose(cand_i, ref_i, rtol=1e-2, atol=1e-2)
    rel = ((cand_i - ref_i).abs().max() / ref_i.abs().max()).item()

    print(f"K=256 (multiple of BLOCK_K=32): allclose={divisible_pass}")
    print(f"K=250 (partial K tile): allclose={irregular_pass}, max rel err={rel:.4f}")
    # What a conventional CI test would conclude: benchmark shapes use round K.
    print(f"NAIVE_ALLCLOSE_VERDICT: {divisible_pass}")
    return divisible_pass and not irregular_pass


if __name__ == "__main__":
    test_kernel()
'''

# --------------------------------------------------------------------------
# FN3-c: per-token INT8 quantization, trailing partial group reuses a scale
# --------------------------------------------------------------------------
INT8_KERNEL = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _group_quant_kernel(X, OUT, stride_xm, stride_om, n_cols, GROUP: tl.constexpr,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = X + row * stride_xm
    o_row = OUT + row * stride_om
    n_groups = n_cols // GROUP

    for g in range(n_groups):
        cols = g * GROUP + tl.arange(0, BLOCK)
        in_group = tl.arange(0, BLOCK) < GROUP
        x = tl.load(x_row + cols, mask=in_group & (cols < n_cols), other=0.0)
        absmax = tl.max(tl.abs(x), axis=0)
        scale = tl.where(absmax == 0.0, 1.0, absmax / 127.0)
        q = tl.floor(x / scale + 0.5).to(tl.int8)
        deq = q.to(tl.float32) * scale
        tl.store(o_row + cols, deq, mask=in_group & (cols < n_cols))


def group_quant_dequant(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Symmetric INT8 quantize-dequantize with one scale per group of columns."""
    n_rows, n_cols = x.shape
    out = torch.zeros_like(x)
    _group_quant_kernel[(n_rows,)](x, out, x.stride(0), out.stride(0), n_cols,
                                   GROUP=group_size, BLOCK=group_size)
    return out
'''

INT8_PROBLEM = """Operation: symmetric INT8 quantize-dequantize with one scale per group of
columns, the per-group scaling used by INT8 inference kernels.

For each group of `group_size` consecutive columns of a row:

    scale = max(|x|) / 127          over that group
    q     = round(x / scale)        clamped into int8
    y     = q * scale

Every column of the input must be quantized under the scale of the group it
belongs to.

Input domain: `n_cols` may be any positive integer. It is NOT required to be a
multiple of `group_size`; when it is not, the trailing columns form a shorter
final group, which is still a group with its own scale.

Does group_quant_dequant() satisfy this contract?
"""

INT8_TEST = '''
import torch
from kernel import group_quant_dequant


def reference(x, group_size):
    out = torch.zeros_like(x)
    n_cols = x.shape[1]
    for start in range(0, n_cols, group_size):
        stop = min(start + group_size, n_cols)
        chunk = x[:, start:stop]
        scale = chunk.abs().amax(dim=1, keepdim=True) / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.round(chunk / scale).clamp(-128, 127)
        out[:, start:stop] = q * scale
    return out


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    group = 64

    # Conventional test shape: n_cols is a multiple of group_size.
    x_r = torch.randn(8, 512, device=device)
    cand_r = group_quant_dequant(x_r, group)
    ref_r = reference(x_r, group)
    divisible_pass = torch.allclose(cand_r, ref_r, rtol=1e-2, atol=1e-2)

    # Contract-permitted irregular shape: a shorter final group exists.
    x_i = torch.randn(8, 500, device=device)
    cand_i = group_quant_dequant(x_i, group)
    ref_i = reference(x_i, group)
    irregular_pass = torch.allclose(cand_i, ref_i, rtol=1e-2, atol=1e-2)
    tail_err = (cand_i[:, 448:] - ref_i[:, 448:]).abs().max().item()

    print(f"n_cols=512 (multiple of group): allclose={divisible_pass}")
    print(f"n_cols=500 (short final group): allclose={irregular_pass}, tail max abs err={tail_err:.4f}")
    # What a conventional CI test would conclude: hidden sizes are round.
    print(f"NAIVE_ALLCLOSE_VERDICT: {divisible_pass}")
    return divisible_pass and not irregular_pass


if __name__ == "__main__":
    test_kernel()
'''

# --------------------------------------------------------------------------
# FN3-d: chunked cumulative scan, chunk loop drops the trailing partial chunk
# --------------------------------------------------------------------------
SCAN_KERNEL = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _chunked_scan_kernel(X, OUT, stride_xb, stride_ob, seqlen, CHUNK: tl.constexpr):
    b = tl.program_id(0)
    x_row = X + b * stride_xb
    o_row = OUT + b * stride_ob
    n_chunks = seqlen // CHUNK

    carry = 0.0
    for c in range(n_chunks):
        offs = c * CHUNK + tl.arange(0, CHUNK)
        x = tl.load(x_row + offs, mask=offs < seqlen, other=0.0)
        local = tl.cumsum(x, axis=0)
        tl.store(o_row + offs, local + carry, mask=offs < seqlen)
        carry += tl.sum(x, axis=0)


def chunked_cumsum(x: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Cumulative sum along the sequence axis, carried across fixed-size chunks."""
    batch, seqlen = x.shape
    out = torch.zeros_like(x)
    _chunked_scan_kernel[(batch,)](x, out, x.stride(0), out.stride(0), seqlen, CHUNK=chunk)
    return out
'''

SCAN_PROBLEM = """Operation: cumulative sum along the sequence axis of a 2-D float32 tensor,
computed chunk by chunk with a running carry (the chunked-scan structure used by
state-space and linear-attention kernels).

    y[b, t] = sum over s <= t of x[b, s]

Input domain: `seqlen` may be any positive integer. It is NOT required to be a
multiple of `chunk`; when it is not, the trailing positions form a shorter final
chunk whose outputs must still be produced.

Does chunked_cumsum() satisfy this contract?
"""

SCAN_TEST = '''
import torch
from kernel import chunked_cumsum


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    chunk = 64

    # Conventional test shape: seqlen is a multiple of the chunk size.
    x_r = torch.randn(4, 512, device=device)
    cand_r = chunked_cumsum(x_r, chunk)
    ref_r = x_r.cumsum(dim=1)
    divisible_pass = torch.allclose(cand_r, ref_r, rtol=1e-2, atol=1e-2)

    # Contract-permitted irregular length: a shorter final chunk exists.
    x_i = torch.randn(4, 500, device=device)
    cand_i = chunked_cumsum(x_i, chunk)
    ref_i = x_i.cumsum(dim=1)
    irregular_pass = torch.allclose(cand_i, ref_i, rtol=1e-2, atol=1e-2)
    tail_err = (cand_i[:, 448:] - ref_i[:, 448:]).abs().max().item()

    print(f"seqlen=512 (multiple of chunk): allclose={divisible_pass}")
    print(f"seqlen=500 (short final chunk): allclose={irregular_pass}, tail max abs err={tail_err:.4f}")
    # What a conventional CI test would conclude: benchmark sequence lengths are round.
    print(f"NAIVE_ALLCLOSE_VERDICT: {divisible_pass}")
    return divisible_pass and not irregular_pass


if __name__ == "__main__":
    test_kernel()
'''

CASES = {
    "fn8_fused_softmax_block_tail": dict(
        kernel=SOFTMAX_KERNEL, problem=SOFTMAX_PROBLEM, test=SOFTMAX_TEST,
        kernel_family="three-pass fused softmax",
        reference="triton-lang/triton, python/tutorials/02-fused-softmax.py; same structure in Liger-Kernel and vLLM normalization kernels",
        mechanism="The normalizing sum runs `n_cols // BLOCK` blocks while the max and store passes run cdiv. When n_cols is a multiple of BLOCK the two agree and the kernel is exact; otherwise the trailing partial block contributes to no denominator, so every entry of the row is scaled by a denominator that is too small and the row sums to about 1.10 instead of 1. Because softmax entries are ~1/n_cols, an atol of 1e-2 swallows the entire error: allclose passes at the irregular shape too, and only the sums-to-one invariant exposes it.",
    ),
    "fn9_tiled_matmul_k_tail_unmasked": dict(
        kernel=MATMUL_KERNEL, problem=MATMUL_PROBLEM, test=MATMUL_TEST,
        kernel_family="tiled GEMM",
        reference="triton-lang/triton, python/tutorials/03-matrix-multiplication.py; the same tiling appears in vLLM and AutoGPTQ matmul kernels",
        mechanism="The K loop runs `K // BLOCK_K` iterations instead of cdiv. When K is a multiple of BLOCK_K the two agree; otherwise the final partial K tile is never accumulated, so the product silently omits the last (K mod BLOCK_K) terms of every dot product.",
    ),
    "fn10_int8_group_quant_short_final_group": dict(
        kernel=INT8_KERNEL, problem=INT8_PROBLEM, test=INT8_TEST,
        kernel_family="per-group symmetric INT8 quantization",
        reference="sgl-project/sglang, python/sglang/srt/layers/quantization/int8_kernel.py; same per-group scaling in AutoGPTQ and llm-awq",
        mechanism="The group loop runs `n_cols // GROUP` times. When group_size divides n_cols every column belongs to a visited group; otherwise the trailing short group is never visited and those columns are left at their initial value instead of being quantized under their own scale.",
    ),
    "fn11_chunked_scan_short_final_chunk": dict(
        kernel=SCAN_KERNEL, problem=SCAN_PROBLEM, test=SCAN_TEST,
        kernel_family="chunked cumulative scan with carry",
        reference="state-spaces/mamba, mamba_ssm/ops/triton/ssd_chunk_scan.py; the same chunk-with-carry structure appears in linear-attention kernels",
        mechanism="The chunk loop runs `seqlen // CHUNK` times, so a sequence length that is not a multiple of the chunk size leaves its trailing positions unwritten. Benchmarks use round sequence lengths, where floor and ceiling division agree.",
    ),
}

META_COMMON = {
    "benchmark_version": "fn_fp_triton_v1",
    "group": "FN",
    "seed_class": "FN3",
    "failure_mode": "false_negative",
    "default_tolerance": {"rtol": 0.01, "atol": 0.01},
    "status": "seed_v2",
    "source": "real_triton_kernel_adapted",
    "requires_gpu": True,
    "passed": None,
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
            **META_COMMON,
            "kernel_family": spec["kernel_family"],
            "reference": spec["reference"],
            "mechanism": spec["mechanism"],
            "expected": {
                "ground_truth": spec["mechanism"].split(".")[0],
                "naive_allclose_verdict": "PASS",
                "correct_verdict": "BUGGY",
            },
        }, indent=2) + "\n")
        print("wrote", name)


if __name__ == "__main__":
    main()
