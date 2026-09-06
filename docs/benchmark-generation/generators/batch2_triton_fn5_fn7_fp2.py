"""Real-Triton-kernel FN/FP cases, sub-batch B: FN5, FN7, FP2.

Sources (fetched & verified 2026-09-04):
  - sgl-project/sglang: python/sglang/kernels/ops/quantization/int8_kernel.py
    (_per_token_quant_int8)
  - linkedin/Liger-Kernel: src/liger_kernel/ops/rms_norm.py
    (_rms_norm_forward_kernel)
  - Dao-AILab/flash-attention: flash_attn/ops/triton/rotary.py (rotary_kernel)

Requires a CUDA GPU (Triton). Use benchmark_fn_fp/triton/modal_runner.py.
"""
import json
import os

DATASET_DIR = os.environ.get("KV_DATASET_DIR", "benchmark_fn_fp/triton")

CASES = []


def add(name, group, seed_class, kernel_family, reference, failure_mode,
        mechanism, expected, note, problem_txt, kernel_py, test_py):
    CASES.append(dict(
        name=name, group=group, seed_class=seed_class, kernel_family=kernel_family,
        reference=reference, failure_mode=failure_mode, mechanism=mechanism,
        expected=expected, note=note, problem_txt=problem_txt, kernel_py=kernel_py,
        test_py=test_py,
    ))


# ---------------------------------------------------------------------------
# FN5: sglang real per-token INT8 quantization kernel, stale-scale bug
# ---------------------------------------------------------------------------
add(
    name="fn5_sglang_int8_quant_stale_scale",
    group="FN", seed_class="FN5",
    kernel_family="sglang per-token INT8 activation quantization",
    reference="sgl-project/sglang, python/sglang/kernels/ops/quantization/int8_kernel.py::_per_token_quant_int8",
    failure_mode="false_negative",
    mechanism="The real kernel computes `absmax = max(max(|x|), 1e-10)` from the CURRENT row "
               "and scales by absmax/127. The bug replaces this with a stale, precomputed "
               "constant (simulating an EMA scale that has not adapted to the current batch). "
               "On mild data whose true absmax happens to equal the stale constant, both "
               "produce bit-identical output; a real outlier value the stale constant hasn't "
               "seen gets severely clipped.",
    expected={
        "ground_truth": "kernel.py replaces the real kernel's per-row tl.max(tl.abs(x)) with a stale hardcoded scale",
        "naive_allclose_verdict": "PASS when current absmax coincides with the stale scale, FAIL once a real outlier appears",
        "correct_verdict": "BUGGY",
    },
    note="kernel.py is the real sglang kernel with one line changed (absmax source). Runs on GPU.",
    problem_txt='''FN5: real sglang per-token INT8 quantization kernel, stale-scale bug.

Source: sgl-project/sglang, python/sglang/kernels/ops/quantization/int8_kernel.py
::_per_token_quant_int8 (used verbatim except for the scale source).

Real: `absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10); scale = absmax / 127`.
Bug: absmax is replaced by a stale constant (as if from an out-of-date EMA),
ignoring the current row's actual magnitude.
''',
    kernel_py='''"""Real sglang per-token INT8 quant kernel, with a stale-scale bug injected.

Verbatim from sgl-project/sglang, python/sglang/kernels/ops/quantization/int8_kernel.py
::_per_token_quant_int8, fetched 2026-09-04, except the `absmax` source.
"""
import torch
import triton
import triton.language as tl

_STALE_ABSMAX = 4.0  # simulates an EMA scale that hasn't caught up to the current batch


@triton.jit
def _per_token_quant_int8(
    x_ptr, xq_ptr, scale_ptr,
    stride_x, stride_xq, N,
    STALE_ABSMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = STALE_ABSMAX  # BUG: real kernel computes tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = tl.clamp(tl.extra.cuda.libdevice.round(x_q), -127, 127).to(tl.int8)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


def quant_dequant_int8(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    x_q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    _per_token_quant_int8[(M,)](x, x_q, scales, x.stride(0), x_q.stride(0), N, STALE_ABSMAX=_STALE_ABSMAX, BLOCK=BLOCK)
    return x_q.float() * scales.unsqueeze(-1)
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import quant_dequant_int8 as quant_dequant_buggy

import triton
import triton.language as tl


@triton.jit
def _per_token_quant_int8_ref(x_ptr, xq_ptr, scale_ptr, stride_x, stride_xq, N, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)  # real: current-row max
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = tl.clamp(tl.extra.cuda.libdevice.round(x_q), -127, 127).to(tl.int8)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


def quant_dequant_int8_reference(x):
    M, N = x.shape
    x_q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    _per_token_quant_int8_ref[(M,)](x, x_q, scales, x.stride(0), x_q.stride(0), N, BLOCK=BLOCK)
    return x_q.float() * scales.unsqueeze(-1)


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    torch.manual_seed(0)

    # Mild: construct the row so its true absmax is EXACTLY the stale constant
    # (4.0), by directly setting one entry to that float value rather than
    # normalizing by division (which can leave the max a few ULPs off from
    # 4.0 and flip a handful of int8 rounding decisions at the bin edge).
    x_mild = torch.randn(4, 1024, device=device) * 0.5
    x_mild[:, 0] = 4.0
    ref_mild = quant_dequant_int8_reference(x_mild)
    cand_mild = quant_dequant_buggy(x_mild)
    mild_pass = torch.allclose(cand_mild, ref_mild, rtol=1e-2, atol=1e-2)

    # Adversarial: one row has a real outlier far beyond the stale scale.
    x_out = torch.randn(4, 1024, device=device)
    x_out[0, 0] = 50.0
    ref_out = quant_dequant_int8_reference(x_out)
    cand_out = quant_dequant_buggy(x_out)
    outlier_pass = torch.allclose(cand_out, ref_out, rtol=1e-2, atol=1e-2)

    print(f"mild (true absmax == stale scale): allclose={mild_pass}")
    print(f"outlier row: allclose={outlier_pass}, reconstructed={cand_out[0,0].item():.2f} vs true {x_out[0,0].item():.2f}")
    print("FN DEMONSTRATED" if mild_pass and not outlier_pass else "tune constants")
    return mild_pass and not outlier_pass


if __name__ == "__main__":
    test_kernel()
''',
)


# ---------------------------------------------------------------------------
# FN7: Liger RMSNorm real kernel, eps placement bug
# ---------------------------------------------------------------------------
add(
    name="fn7_liger_rmsnorm_eps_placement",
    group="FN", seed_class="FN7",
    kernel_family="Liger-Kernel fused RMSNorm forward",
    reference="linkedin/Liger-Kernel, src/liger_kernel/ops/rms_norm.py::_rms_norm_forward_kernel",
    failure_mode="false_negative",
    mechanism="The real kernel computes `rstd = rsqrt(mean_square + eps)`. The bug moves eps "
               "outside the sqrt: `rstd = 1 / (sqrt(mean_square) + eps)`. At ordinary "
               "magnitudes the two formulas are numerically indistinguishable; only a "
               "near-all-zero row (padding, pruned embeddings) makes the two denominators "
               "differ by orders of magnitude.",
    expected={
        "ground_truth": "kernel.py moves eps outside the sqrt in the real Liger RMSNorm kernel",
        "naive_allclose_verdict": "PASS on ordinary rows, FAIL on a near-zero row",
        "correct_verdict": "BUGGY",
    },
    note="kernel.py is the real Liger-Kernel forward kernel with one line changed (rstd formula). Runs on GPU.",
    problem_txt='''FN7: real Liger-Kernel RMSNorm forward kernel, eps-placement bug.

Source: linkedin/Liger-Kernel, src/liger_kernel/ops/rms_norm.py
::_rms_norm_forward_kernel (used verbatim except for the rstd formula).

Real: rstd = rsqrt(mean_square + eps).
Bug:  rstd = 1 / (sqrt(mean_square) + eps).
''',
    kernel_py='''"""Real Liger-Kernel RMSNorm forward kernel, with an eps-placement bug injected.

Verbatim (casting-mode logic simplified to CASTING_MODE_NONE only) from
linkedin/Liger-Kernel, src/liger_kernel/ops/rms_norm.py::_rms_norm_forward_kernel,
fetched 2026-09-04, except the rstd formula.
"""
import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.libdevice import sqrt as _sqrt
except ModuleNotFoundError:
    from triton.language.extra.cuda.libdevice import sqrt as _sqrt


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr, Y_row_stride, X_ptr, X_row_stride, RSTD_ptr, RSTD_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0)
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = 1.0 / (_sqrt(mean_square) + eps)  # BUG: real kernel does rsqrt(mean_square + eps)

    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd)
    Y_row = X_row * rstd
    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row, mask=mask)


def rms_norm_forward(X: torch.Tensor, eps: float) -> torch.Tensor:
    num_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    Y = torch.empty_like(X)
    RSTD = torch.empty(num_rows, dtype=torch.float32, device=X.device)
    _rms_norm_forward_kernel[(num_rows,)](Y, Y.stride(0), X, X.stride(0), RSTD, RSTD.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return Y
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import rms_norm_forward as rms_norm_buggy

import triton
import triton.language as tl

try:
    from triton.language.extra.libdevice import rsqrt
except ModuleNotFoundError:
    from triton.language.extra.cuda.libdevice import rsqrt


@triton.jit
def _rms_norm_forward_kernel_ref(
    Y_ptr, Y_row_stride, X_ptr, X_row_stride, RSTD_ptr, RSTD_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0)
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)  # real kernel: eps inside the sqrt
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd)
    Y_row = X_row * rstd
    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row, mask=mask)


def rms_norm_forward_reference(X, eps):
    num_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    Y = torch.empty_like(X)
    RSTD = torch.empty(num_rows, dtype=torch.float32, device=X.device)
    _rms_norm_forward_kernel_ref[(num_rows,)](Y, Y.stride(0), X, X.stride(0), RSTD, RSTD.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return Y


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    eps = 1e-6

    x_normal = torch.randn(8, 16, device=device)
    ref_normal = rms_norm_forward_reference(x_normal, eps)
    cand_normal = rms_norm_buggy(x_normal, eps)
    normal_pass = torch.allclose(cand_normal, ref_normal, rtol=1e-2, atol=1e-2)

    x_tiny = torch.full((1, 16), 2e-8, device=device)
    ref_tiny = rms_norm_forward_reference(x_tiny, eps)
    cand_tiny = rms_norm_buggy(x_tiny, eps)
    tiny_pass = torch.allclose(cand_tiny, ref_tiny, rtol=1e-2, atol=1e-2)

    print(f"normal rows: allclose={normal_pass}")
    print(f"near-zero row: allclose={tiny_pass}, ref={ref_tiny.abs().max().item():.3e}, cand={cand_tiny.abs().max().item():.3e}")
    print("FN DEMONSTRATED" if normal_pass and not tiny_pass else "tune constants")
    return normal_pass and not tiny_pass


if __name__ == "__main__":
    test_kernel()
''',
)


# ---------------------------------------------------------------------------
# FP2: flash-attention real RoPE kernel, both conventions in one kernel
# ---------------------------------------------------------------------------
add(
    name="fp2_flashattn_rope_neox_vs_gptj",
    group="FP", seed_class="FP2",
    kernel_family="flash-attention RoPE kernel (both layout conventions)",
    reference="Dao-AILab/flash-attention, flash_attn/ops/triton/rotary.py::rotary_kernel",
    failure_mode="false_positive",
    mechanism="The real flash-attention rotary kernel implements BOTH RoPE conventions in one "
               "file via the INTERLEAVED flag: split-half (GPT-NeoX) and interleaved-pairs "
               "(GPT-J). Both are mathematically valid as long as q and k use the SAME "
               "convention consistently. Comparing the raw rotated tensors element-by-element "
               "across conventions fails almost everywhere, even though the resulting "
               "attention scores agree once the same permutation is applied to both q and k.",
    expected={
        "ground_truth": "kernel.py is the real flash-attention kernel called with a different (but equally valid) INTERLEAVED convention",
        "naive_allclose_verdict": "FAIL on raw rotated tensors",
        "correct_verdict": "CORRECT (downstream attention scores match within tolerance)",
    },
    note="kernel.py is the real, unmodified flash-attention rotary_kernel (simplified to drop "
         "varlen/GQA-broadcast features unrelated to this mechanism). Runs on GPU.",
    problem_txt='''FP2: real flash-attention RoPE kernel, both layout conventions.

Source: Dao-AILab/flash-attention, flash_attn/ops/triton/rotary.py::rotary_kernel
(used verbatim; varlen/seqlen-offset/conjugate features unrelated to this
mechanism are fixed off for a minimal standalone kernel).

The same real kernel supports both the GPT-NeoX (split-half, INTERLEAVED=False)
and GPT-J (interleaved-pairs, INTERLEAVED=True) conventions. Both are valid;
raw outputs differ elementwise, but attention scores computed consistently
within either convention agree.
''',
    kernel_py='''"""Real flash-attention RoPE kernel (both conventions), simplified to drop
varlen/seqlen-offset/conjugate features unrelated to this mechanism.

Verbatim core math from Dao-AILab/flash-attention, flash_attn/ops/triton/rotary.py
::rotary_kernel, fetched 2026-09-04.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    OUT, X, COS, SIN,
    seqlen, nheads, rotary_dim_half,
    stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    INTERLEAVED: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_head = tl.program_id(0)
    pid_m = tl.program_id(1)
    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_half = tl.arange(0, BLOCK_K // 2)

    COS = COS + (rm[:, None] * rotary_dim_half + rk_half[None, :])
    SIN = SIN + (rm[:, None] * rotary_dim_half + rk_half[None, :])
    mask_cs = (rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)

    if not INTERLEAVED:
        X_ptrs = X + (rh[:, None, None] * stride_x_nheads + rm[None, :, None] * stride_x_seqlen + rk_half[None, None, :] * stride_x_headdim)
        OUT_ptrs = OUT + (rh[:, None, None] * stride_out_nheads + rm[None, :, None] * stride_out_seqlen + rk_half[None, None, :] * stride_out_headdim)
        mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk_half[None, None, :] < rotary_dim_half)
        x0 = tl.load(X_ptrs, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(X_ptrs + rotary_dim_half * stride_x_headdim, mask=mask, other=0.0).to(tl.float32)
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT_ptrs, o0, mask=mask)
        tl.store(OUT_ptrs + rotary_dim_half * stride_out_headdim, o1, mask=mask)
    else:
        rk = tl.arange(0, BLOCK_K)
        X_ptrs = X + (rh[:, None, None] * stride_x_nheads + rm[None, :, None] * stride_x_seqlen + rk[None, None, :] * stride_x_headdim)
        OUT_ptrs = OUT + (rh[:, None, None] * stride_out_nheads + rm[None, :, None] * stride_out_seqlen + rk[None, None, :] * stride_out_headdim)
        mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk[None, None, :] < 2 * rotary_dim_half)
        x = tl.load(X_ptrs, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
        tl.store(OUT_ptrs, o, mask=mask)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """x: (seqlen, nheads, headdim). cos/sin: (seqlen, headdim//2)."""
    seqlen, nheads, headdim = x.shape
    rotary_dim_half = headdim // 2
    out = torch.empty_like(x)
    BLOCK_K = triton.next_power_of_2(headdim)
    grid = (nheads, seqlen)
    rotary_kernel[grid](
        out, x, cos, sin,
        seqlen, nheads, rotary_dim_half,
        out.stride(0), out.stride(1), out.stride(2),
        x.stride(0), x.stride(1), x.stride(2),
        INTERLEAVED=interleaved, BLOCK_H=1, BLOCK_M=1, BLOCK_K=BLOCK_K,
    )
    return out
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import apply_rotary  # real flash-attention rotary kernel, both conventions


def make_cos_sin(seq_len, half, base=10000.0, device="cuda"):
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def to_interleaved(x, half):
    out = torch.empty_like(x)
    out[..., 0::2] = x[..., :half]
    out[..., 1::2] = x[..., half:]
    return out


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    torch.manual_seed(0)
    seq_len, nheads, headdim = 4, 2, 8
    half = headdim // 2

    q = torch.randn(seq_len, nheads, headdim, device=device)
    k = torch.randn(seq_len, nheads, headdim, device=device)
    cos, sin = make_cos_sin(seq_len, half, device=device)

    q_ref = apply_rotary(q, cos, sin, interleaved=False)
    k_ref = apply_rotary(k, cos, sin, interleaved=False)

    q_cand = apply_rotary(to_interleaved(q, half), cos, sin, interleaved=True)
    k_cand = apply_rotary(to_interleaved(k, half), cos, sin, interleaved=True)

    raw_pass = torch.allclose(q_cand, q_ref, rtol=1e-2, atol=1e-2)

    scores_ref = torch.einsum("shd,thd->hst", q_ref, k_ref)
    scores_cand = torch.einsum("shd,thd->hst", q_cand, k_cand)
    downstream_pass = torch.allclose(scores_cand, scores_ref, rtol=1e-2, atol=1e-2)

    print(f"real flash-attention rope kernel (GPU): raw rotated-tensor allclose = {raw_pass} (expected False)")
    print(f"downstream attention-score allclose = {downstream_pass} (expected True)")
    print("FP DEMONSTRATED" if (not raw_pass) and downstream_pass else "tune constants")
    return (not raw_pass) and downstream_pass


if __name__ == "__main__":
    test_kernel()
''',
)


def write_case(case, dataset_dir):
    entry_dir = os.path.join(dataset_dir, case["name"])
    os.makedirs(entry_dir, exist_ok=True)
    with open(os.path.join(entry_dir, "problem.txt"), "w") as f:
        f.write(case["problem_txt"].strip() + "\n")
    with open(os.path.join(entry_dir, "kernel.py"), "w") as f:
        f.write(case["kernel_py"])
    with open(os.path.join(entry_dir, "test.py"), "w") as f:
        f.write(case["test_py"])
    meta = {
        "name": case["name"],
        "benchmark_version": "fn_fp_triton_v1",
        "group": case["group"],
        "seed_class": case["seed_class"],
        "kernel_family": case["kernel_family"],
        "reference": case["reference"],
        "failure_mode": case["failure_mode"],
        "mechanism": case["mechanism"],
        "default_tolerance": {"rtol": 0.01, "atol": 0.01},
        "expected": case["expected"],
        "note": case["note"],
        "status": "seed_v1",
        "source": "real_triton_kernel_adapted",
        "requires_gpu": True,
        "passed": None,
    }
    with open(os.path.join(entry_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def main():
    for case in CASES:
        write_case(case, DATASET_DIR)
    print(f"wrote {len(CASES)} cases to {os.path.abspath(DATASET_DIR)}")
    for case in CASES:
        print(f"  {case['group']:2s} {case['seed_class']:4s} {case['name']}")


if __name__ == "__main__":
    main()
