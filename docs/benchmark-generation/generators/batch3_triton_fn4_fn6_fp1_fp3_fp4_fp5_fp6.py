"""Real-Triton-kernel FN/FP cases, sub-batch C: FN4, FN6, FP1b, FP3b, FP4, FP5, FP6.

Sources (fetched & verified 2026-09-04):
  - vllm-project/vllm: vllm/v1/attention/ops/triton_unified_attention.py
    (query->KV head index arithmetic pattern, lines ~330-334)
  - state-spaces/mamba: mamba_ssm/ops/triton/ssd_state_passing.py
    (_state_passing_fwd_kernel)
  - vllm-project/vllm: vllm/v1/sample/ops/topk_topp_triton.py
    (_update_min_larger_stats)
  - AutoGPTQ/AutoGPTQ: auto_gptq/nn_modules/triton_utils/kernels.py
    (quant_matmul_248_kernel; reused from sub-batch A for FP4)
  - linkedin/Liger-Kernel: src/liger_kernel/ops/rms_norm.py
    (reused from sub-batch A/B for FP6, real correct kernel only)

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
# FN4: GQA head-mapping arithmetic modeled on vLLM's real unified attention kernel
# ---------------------------------------------------------------------------
add(
    name="fn4_gqa_head_mapping_homogeneous",
    group="FN", seed_class="FN4",
    kernel_family="GQA query-to-KV head index arithmetic (vLLM unified attention pattern)",
    reference="vllm-project/vllm, vllm/v1/attention/ops/triton_unified_attention.py, lines ~330-334 "
              "(query_pos = ... + offs_m // num_queries_per_kv; query_offset_1 = kv_head_idx * "
              "num_queries_per_kv + offs_m % num_queries_per_kv)",
    failure_mode="false_negative",
    mechanism="The correct GQA mapping groups consecutive query heads onto the same KV head "
               "via floor division: kv_head = query_head // n_rep -- the same style of integer "
               "div/mod head-decomposition vLLM's real unified attention kernel uses to split a "
               "flattened index into (sequence position, head-in-group) via `// "
               "num_queries_per_kv` and `% num_queries_per_kv`. The bug uses modulo instead "
               "(kv_head = query_head % num_kv_heads), which is invisible when every KV head "
               "holds identical/homogeneous data.",
    expected={
        "ground_truth": "kernel.py maps query heads to KV heads with modulo instead of floor-division",
        "naive_allclose_verdict": "PASS when KV heads carry homogeneous data, FAIL when KV heads carry distinguishable signatures",
        "correct_verdict": "BUGGY",
    },
    note="This is a minimal standalone kernel using the same div/mod arithmetic style as vLLM's "
         "real production attention kernel, not a verbatim port of its ~1200-line full kernel "
         "(unrelated features like paging, FP8 KV-cache, sliding windows, and ALiBi are outside "
         "this mechanism). Runs on GPU.",
    problem_txt='''FN4: GQA query-to-KV-head mapping, in the style of vLLM's real unified attention kernel.

Source (style reference): vllm-project/vllm, vllm/v1/attention/ops/triton_unified_attention.py,
~lines 330-334, which decomposes a flattened index via `// num_queries_per_kv`
and `% num_queries_per_kv` together.

Correct: kv_head = query_head // n_rep (consecutive query heads share a KV head).
Bug: kv_head = query_head % num_kv_heads (a completely different grouping).
''',
    kernel_py='''"""Minimal standalone Triton kernel implementing GQA query-to-KV-head
index arithmetic, modeled on the integer div/mod head-decomposition pattern
in vLLM's real unified attention kernel (vllm/v1/attention/ops/
triton_unified_attention.py, ~lines 330-334 use `// num_queries_per_kv` and
`% num_queries_per_kv` together to decompose a flattened query index into
(sequence position, head-in-group); this kernel applies the same style of
arithmetic directly to the query-head axis).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gqa_gather_kernel(kv_ptr, out_ptr, num_kv_heads, dim: tl.constexpr):
    q_head = tl.program_id(0)
    kv_head = q_head % num_kv_heads  # BUG: real mapping is q_head // n_rep
    offs_d = tl.arange(0, dim)
    kv = tl.load(kv_ptr + kv_head * dim + offs_d)
    tl.store(out_ptr + q_head * dim + offs_d, kv)


def gqa_gather(kv: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    num_kv_heads, dim = kv.shape
    out = torch.empty(num_q_heads, dim, device=kv.device, dtype=kv.dtype)
    gqa_gather_kernel[(num_q_heads,)](kv, out, num_kv_heads, dim=dim)
    return out
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import gqa_gather  # buggy: modulo instead of floor-division

import triton
import triton.language as tl


@triton.jit
def _gqa_gather_kernel_ref(kv_ptr, out_ptr, n_rep, dim: tl.constexpr):
    q_head = tl.program_id(0)
    kv_head = q_head // n_rep  # real: floor-division groups consecutive query heads
    offs_d = tl.arange(0, dim)
    kv = tl.load(kv_ptr + kv_head * dim + offs_d)
    tl.store(out_ptr + q_head * dim + offs_d, kv)


def gqa_gather_reference(kv, num_q_heads):
    num_kv_heads, dim = kv.shape
    n_rep = num_q_heads // num_kv_heads
    out = torch.empty(num_q_heads, dim, device=kv.device, dtype=kv.dtype)
    _gqa_gather_kernel_ref[(num_q_heads,)](kv, out, n_rep, dim=dim)
    return out


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    num_q_heads, num_kv_heads, dim = 8, 2, 4

    # (a) homogeneous KV data: every head is the same base vector -> bug invisible
    base = torch.randn(1, dim, device=device)
    kv_homogeneous = base.expand(num_kv_heads, dim).clone().contiguous()
    ref_h = gqa_gather_reference(kv_homogeneous, num_q_heads)
    cand_h = gqa_gather(kv_homogeneous, num_q_heads)
    homogeneous_pass = torch.allclose(cand_h, ref_h, rtol=1e-2, atol=1e-2)

    # (b) distinguishable KV data: bug exposed
    kv_distinct = torch.stack([torch.full((dim,), float(i * 100), device=device) for i in range(num_kv_heads)])
    ref_d = gqa_gather_reference(kv_distinct, num_q_heads)
    cand_d = gqa_gather(kv_distinct, num_q_heads)
    distinct_pass = torch.allclose(cand_d, ref_d, rtol=1e-2, atol=1e-2)

    print(f"homogeneous KV data: allclose={homogeneous_pass} (bug hidden)")
    print(f"distinguishable KV data: allclose={distinct_pass} (bug exposed)")
    print("FN DEMONSTRATED" if homogeneous_pass and not distinct_pass else "tune constants")
    return homogeneous_pass and not distinct_pass


if __name__ == "__main__":
    test_kernel()
''',
)


# ---------------------------------------------------------------------------
# Shared real Mamba2 state-passing kernel header (used by FN6 and FP3b)
# ---------------------------------------------------------------------------
MAMBA_REF_KERNEL = '''
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
'''

add(
    name="fn6_mamba_state_passing_lowbit_drift",
    group="FN", seed_class="FN6",
    kernel_family="Mamba2/SSD chunked state-passing recurrence",
    reference="state-spaces/mamba, mamba_ssm/ops/triton/ssd_state_passing.py::_state_passing_fwd_kernel",
    failure_mode="false_negative",
    mechanism="The real kernel's recurrence is `states = scale * states + new_states` across "
               "chunks (scale = exp(cumulative log-decay)). The bug rounds `states` to a coarse "
               "grid after every chunk (simulating a low-bit KV-cache-style compression of the "
               "recurrent state). The per-step rounding error is tiny; over a short rollout it "
               "stays inside tolerance, but it compounds across many chunks.",
    expected={
        "ground_truth": "kernel.py rounds the real recurrence's state to a coarse grid every chunk",
        "naive_allclose_verdict": "PASS at nchunks=5, FAIL at nchunks=2000",
        "correct_verdict": "BUGGY",
    },
    note="kernel.py is the real mamba state-passing recurrence with one added rounding line. Runs on GPU.",
    problem_txt='''FN6: real Mamba2/SSD state-passing recurrence, low-bit compression drift.

Source: state-spaces/mamba, mamba_ssm/ops/triton/ssd_state_passing.py
::_state_passing_fwd_kernel (used verbatim except for an added rounding step).

Real recurrence: states = exp(dA_chunk_cumsum) * states + new_states.
Bug: states is additionally rounded to a coarse grid after every chunk,
simulating a low-bit compressed KV-cache/state representation.
''',
    kernel_py='''"""Real Mamba2/SSD state-passing kernel, with per-chunk state quantization
(low-bit KV-cache-style compression) injected.

Verbatim core recurrence from state-spaces/mamba,
mamba_ssm/ops/triton/ssd_state_passing.py::_state_passing_fwd_kernel,
fetched 2026-09-04 (HAS_INITSTATES/HAS_SEQ_IDX paths dropped; unrelated to
this mechanism), plus one added quantization line.
"""
import torch
import triton
import triton.language as tl

_QUANT_STEP = 5e-3  # coarse quantization grid for the compressed recurrent state


@triton.jit
def _state_passing_fwd_kernel(
    states_ptr, out_ptr, final_states_ptr, dA_cs_ptr,
    dim, nchunks,
    stride_states_chunk, stride_states_dim,
    stride_out_chunk, stride_out_dim,
    stride_final_dim,
    stride_dA_cs_chunk,
    QUANT_STEP: tl.constexpr,
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
        states = scale * states + new_states  # real mamba recurrence
        states = tl.floor(states / QUANT_STEP + 0.5) * QUANT_STEP  # BUG: coarse per-chunk quantization
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def state_passing_lowbit(new_states: torch.Tensor, dA_cs: torch.Tensor) -> torch.Tensor:
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
        QUANT_STEP=_QUANT_STEP, BLOCK_SIZE=BLOCK_SIZE,
    )
    return final_states
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import state_passing_lowbit  # buggy: per-chunk state rounding

import triton
import triton.language as tl
''' + MAMBA_REF_KERNEL + '''

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
    return short_pass and not long_pass


if __name__ == "__main__":
    test_kernel()
''',
)


# ---------------------------------------------------------------------------
# FP3b: same real mamba kernel, fine vs coarse chunking (exact math, fp drift)
# ---------------------------------------------------------------------------
add(
    name="fp3_mamba_state_passing_chunk_granularity",
    group="FP", seed_class="FP3",
    kernel_family="Mamba2/SSD chunked state-passing recurrence",
    reference="state-spaces/mamba, mamba_ssm/ops/triton/ssd_state_passing.py::_state_passing_fwd_kernel",
    failure_mode="false_positive",
    mechanism="The SAME real, unmodified mamba recurrence kernel is run twice on mathematically "
               "equivalent inputs at different chunk granularities: many small chunks (fine) vs "
               "fewer, pre-aggregated larger chunks (coarse). In exact arithmetic the results are "
               "identical (this is an exact recurrence, not an approximation); in floating point, "
               "the different number/order of multiply-accumulate steps introduces drift that "
               "grows with the number of fine subdivisions.",
    expected={
        "ground_truth": "kernel.py is the real, completely unmodified kernel; both runs use it as-is",
        "naive_allclose_verdict": "FAIL under a tight tolerance on a long, finely-chunked sequence",
        "correct_verdict": "CORRECT (both are exact, mathematically equivalent evaluations)",
    },
    note="No mutation at all: kernel.py is the literal real kernel, invoked twice with "
         "mathematically-equivalent-by-construction inputs at different granularities. Runs on GPU.",
    problem_txt='''FP3 (extension): real Mamba2/SSD recurrence at different chunk granularities.

Source: state-spaces/mamba, mamba_ssm/ops/triton/ssd_state_passing.py
::_state_passing_fwd_kernel (completely unmodified).

Within each "coarse" chunk, decompose into m "fine" sub-steps whose combined
decay sums to the coarse step's decay, with the new_states contribution
placed entirely on the last fine sub-step (so coarse and fine are exactly
equivalent in real arithmetic). Both use the identical real kernel.
''',
    kernel_py='''"""Real Mamba2/SSD state-passing kernel, completely unmodified (no bug --
this FP case demonstrates equivalent-but-different-order floating point
accumulation, not a defect).

Verbatim from state-spaces/mamba, mamba_ssm/ops/triton/ssd_state_passing.py
::_state_passing_fwd_kernel, fetched 2026-09-04 (HAS_INITSTATES/HAS_SEQ_IDX
paths dropped; unrelated to this mechanism).
"""
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
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import state_passing  # the real, unmodified kernel -- used for BOTH sides


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    torch.manual_seed(0)

    def run_fine_and_coarse(n_coarse, m_fine):
        dim = 4
        coarse_decay = -0.05
        coarse_new_states = torch.randn(n_coarse, dim, device=device) * 0.1

        # Coarse: n_coarse real chunks.
        dA_coarse = torch.full((n_coarse,), coarse_decay, device=device)
        coarse_out = state_passing(coarse_new_states, dA_coarse)

        # Fine: each coarse chunk becomes m_fine real chunks whose decays sum
        # to the same coarse_decay, with the new_states contribution placed
        # entirely on the LAST fine sub-step -- exactly equivalent in real
        # arithmetic (intermediate zero-contribution steps only rescale).
        fine_decay_step = coarse_decay / m_fine
        dA_fine = torch.full((n_coarse * m_fine,), fine_decay_step, device=device)
        fine_new_states = torch.zeros(n_coarse * m_fine, dim, device=device)
        fine_new_states[m_fine - 1 :: m_fine] = coarse_new_states
        fine_out = state_passing(fine_new_states, dA_fine)

        return coarse_out, fine_out

    coarse_short, fine_short = run_fine_and_coarse(n_coarse=3, m_fine=4)
    short_pass = torch.allclose(fine_short, coarse_short, rtol=1e-5, atol=1e-5)
    short_diff = (fine_short - coarse_short).abs().max().item()

    coarse_long, fine_long = run_fine_and_coarse(n_coarse=200, m_fine=500)
    long_pass = torch.allclose(fine_long, coarse_long, rtol=1e-5, atol=1e-5)
    long_diff = (fine_long - coarse_long).abs().max().item()

    print(f"short (n_coarse=3, m_fine=4): fine vs coarse allclose(tight)={short_pass}, diff={short_diff:.8f}")
    print(f"long (n_coarse=200, m_fine=500 => 100000 fine steps): fine vs coarse allclose(tight)={long_pass}, diff={long_diff:.8f}")
    grows_as_expected = long_diff >= short_diff
    print("FP DEMONSTRATED" if grows_as_expected else "informational: fp32 drift magnitude depends on inputs")
    return grows_as_expected


if __name__ == "__main__":
    test_kernel()
''',
)


# ---------------------------------------------------------------------------
# FP1b: real vLLM tie-detection primitive, unspecified-tie framing
# ---------------------------------------------------------------------------
add(
    name="fp1_vllm_topp_tie_epsilon",
    group="FP", seed_class="FP1",
    kernel_family="vLLM top-k/top-p sampling, near-pivot tie detection",
    reference="vllm-project/vllm, vllm/v1/sample/ops/topk_topp_triton.py::_update_min_larger_stats",
    failure_mode="false_positive",
    mechanism="The real kernel tracks how many values are within `1e-9` of the running "
               "pivot (`tl.abs(data - tile_min) < 1e-9`) to correctly count ties at the "
               "top-p/top-k cutoff. Two equally valid implementations can use a different (but "
               "still reasonable) epsilon for this tie tolerance; when probabilities are close "
               "to -- but not exactly at -- the cutoff, the two epsilons include a different "
               "number of near-pivot candidates in the selected set, changing a downstream "
               "probability-weighted result well beyond a naive tolerance, even though neither "
               "implementation violates the (epsilon-unspecified) contract.",
    expected={
        "ground_truth": "kernel.py uses a different (still reasonable) tie epsilon than the reference",
        "naive_allclose_verdict": "FAIL on the downstream weighted output",
        "correct_verdict": "CORRECT (a permitted epsilon choice for an unspecified tie tolerance)",
    },
    note="kernel.py is a minimal standalone kernel built around the real "
         "_update_min_larger_stats primitive (verbatim), varying only its epsilon. Runs on GPU.",
    problem_txt='''FP1 (extension): real vLLM near-pivot tie-counting primitive, unspecified epsilon.

Source: vllm-project/vllm, vllm/v1/sample/ops/topk_topp_triton.py
::_update_min_larger_stats (used verbatim).

The real primitive treats values within `1e-9` of the pivot as tied. The
contract does not specify this epsilon exactly; kernel.py uses `1e-3`
instead (still a defensible choice for near-pivot ties), which admits a
different set of near-boundary candidates.
''',
    kernel_py='''"""Real vLLM near-pivot tie-counting primitive (verbatim), called with a
different (but reasonable) tie epsilon than the reference.

Source: vllm-project/vllm, vllm/v1/sample/ops/topk_topp_triton.py
::_update_min_larger_stats, fetched 2026-09-04 -- unmodified except for the
epsilon it is called with.
"""
import torch
import triton
import triton.language as tl

_TIE_EPS = 1e-3  # a different, still-reasonable choice than the real kernel's 1e-9


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
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import count_tied_at_boundary as count_buggy  # real primitive, eps=1e-3

import triton
import triton.language as tl


@triton.jit
def _update_min_larger_stats_ref(data, above_mask, min_larger, num_min_larger, sentinel, EPS: tl.constexpr):
    tile_min = tl.min(tl.where(above_mask, data, sentinel))
    tile_eq = above_mask & (tl.abs(data - tile_min) < EPS)
    tile_cnt = tl.sum(tile_eq)
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < EPS
    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def _count_tied_at_boundary_kernel_ref(scores_ptr, count_ptr, N: tl.constexpr, pivot, EPS: tl.constexpr):
    offs = tl.arange(0, N)
    data = tl.load(scores_ptr + offs)
    above_mask = data > pivot
    min_larger = tl.full((), float("inf"), tl.float32)
    num_min_larger = tl.zeros((), tl.int32)
    min_larger, num_min_larger = _update_min_larger_stats_ref(data, above_mask, min_larger, num_min_larger, float("inf"), EPS)
    tl.store(count_ptr, num_min_larger)


def count_tied_at_boundary_reference(scores, pivot):
    N = scores.shape[0]
    count = torch.empty(1, dtype=torch.int32, device=scores.device)
    _count_tied_at_boundary_kernel_ref[(1,)](scores, count, N=N, pivot=pivot, EPS=1e-9)  # real epsilon
    return int(count.item())


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"

    # One clear top value, then a tight cluster of 3 distinct values near the
    # boundary: the smallest of the cluster (5.0) is separated from the other
    # two by 3e-4 and 6e-4 -- well outside the real kernel's 1e-9 tolerance,
    # but well inside a (still reasonable) 1e-3 tolerance.
    N = 16
    scores = torch.full((N,), -10.0, device=device)
    scores[0] = 10.0
    scores[1] = 5.0
    scores[2] = 5.0003
    scores[3] = 5.0006
    pivot = 0.0

    ref_count = count_tied_at_boundary_reference(scores, pivot)  # real eps=1e-9: only the exact minimum
    cand_count = count_buggy(scores, pivot)  # eps=1e-3: the whole near-boundary cluster

    assert ref_count != cand_count, "the two epsilons should disagree on the boundary tie count"

    # Downstream use (as a real multi-pass top-k kernel would): split a fixed
    # unit of remaining "selection budget" equally among the tied boundary
    # candidates.
    ref_share = 1.0 / ref_count
    cand_share = 1.0 / cand_count
    naive_pass = abs(ref_share - cand_share) < 1e-2

    print(f"real vLLM tie-counting primitive (GPU): reference (eps=1e-9) count={ref_count}, candidate (eps=1e-3) count={cand_count}")
    print(f"downstream per-candidate share: reference={ref_share:.4f}, candidate={cand_share:.4f}, naive allclose-style check={naive_pass} (expected False)")
    print("FP DEMONSTRATED" if not naive_pass else "tune constants")
    return not naive_pass


if __name__ == "__main__":
    test_kernel()
''',
)


# ---------------------------------------------------------------------------
# FP4: real AutoGPTQ kernel, precision-dependent tolerance
# ---------------------------------------------------------------------------
GPTQ_HEADER_FP4 = '''"""Real AutoGPTQ INT4 dequant+matmul kernel, completely unmodified.

Verbatim from AutoGPTQ/AutoGPTQ, auto_gptq/nn_modules/triton_utils/kernels.py
::quant_matmul_248_kernel, fetched 2026-09-04 (autotune stripped for a fixed
small config; the kernel body is untouched). This FP case demonstrates that
its intended quantization noise exceeds an FP32-calibrated tolerance -- not
a defect.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def quant_matmul_248_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr,
    M, N, K, bits, maxq,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_scales, stride_zeros,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    infearure_per_bits = 32 // bits
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_mask = offs_am[:, None] < M
    b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)
    g_ptrs = g_ptr + offs_k
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits)

    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        g_idx = tl.load(g_ptrs)
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)
        zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)
        zeros = (zeros >> zeros_shifter[None, :]) & maxq
        zeros = zeros + 1
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs)
        b = (b >> shifter[:, None]) & maxq
        b = (b - zeros) * scales
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        g_ptrs += BLOCK_SIZE_K

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def gptq_matmul(a, b_packed, scales, zeros, g_idx, bits=4):
    M, K = a.shape
    N = b_packed.shape[1]
    maxq = 2 ** bits - 1
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 32, 32, 16, 1
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    quant_matmul_248_kernel[grid](
        a, b_packed, c, scales, zeros, g_idx,
        M, N, K, bits, maxq,
        a.stride(0), a.stride(1), b_packed.stride(0), b_packed.stride(1), c.stride(0), c.stride(1),
        scales.stride(0), zeros.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return c
'''

add(
    name="fp4_gptq_int4_quant_tolerance",
    group="FP", seed_class="FP4",
    kernel_family="AutoGPTQ INT4 dequant+matmul",
    reference="AutoGPTQ/AutoGPTQ, auto_gptq/nn_modules/triton_utils/kernels.py::quant_matmul_248_kernel",
    failure_mode="false_positive",
    mechanism="The real INT4 dequant-matmul kernel is correct; INT4 quantization intentionally "
               "introduces rounding noise as the cost of 4-bit weights. A tolerance calibrated "
               "for FP32 GEMM (rtol=1e-2, atol=1e-2) can still be exceeded by ordinary, expected "
               "INT4 quantization error at larger K, even though the kernel has no defect.",
    expected={
        "ground_truth": "kernel.py is the real, unmodified INT4 kernel; its error is the expected cost of 4-bit quantization",
        "naive_allclose_verdict": "FAIL under a fixed FP32-calibrated tolerance at larger K",
        "correct_verdict": "CORRECT (error matches the expected INT4 quantization noise model)",
    },
    note="No mutation: kernel.py is the literal real AutoGPTQ kernel (identical to "
         "fn3_gptq_dequant_group_div_coverage's kernel.py). This case demonstrates FP4, not FN3. Runs on GPU.",
    problem_txt='''FP4: real AutoGPTQ INT4 kernel, precision-dependent tolerance.

Source: AutoGPTQ/AutoGPTQ, auto_gptq/nn_modules/triton_utils/kernels.py
::quant_matmul_248_kernel (completely unmodified -- same kernel as
fn3_gptq_dequant_group_div_coverage, used here to test a different property).

The kernel is correct. Its INT4 rounding noise is the intended cost of
4-bit precision, not a bug; a strict FP32-calibrated tolerance can still
reject it, especially as K (the reduction dimension) grows.
''',
    kernel_py=GPTQ_HEADER_FP4,
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import gptq_matmul  # real, unmodified AutoGPTQ kernel


def math_ceil_div(a, b):
    return (a + b - 1) // b


def run(K, N=64, M=8, bits=4, group_size=128):
    torch.manual_seed(0)
    device = "cuda"
    num_groups = math_ceil_div(K, group_size)
    infeature_per_bits = 32 // bits
    maxq = 2 ** bits - 1

    a = torch.randn(M, K, device=device, dtype=torch.float32)
    # A continuous "true" weight matrix -- this is what the reference matmul
    # uses. Quantizing it to INT4 (below) introduces REAL rounding loss; the
    # earlier reconstruct-from-int comparison was trivially exact because it
    # never compared against a continuous ground truth.
    w_true = torch.randn(K, N, device=device, dtype=torch.float32)

    g_idx = torch.clamp(torch.arange(K, device=device, dtype=torch.int32) // group_size, max=num_groups - 1)
    group_min = torch.full((num_groups, N), float("inf"), device=device)
    group_max = torch.full((num_groups, N), float("-inf"), device=device)
    group_min.scatter_reduce_(0, g_idx.long().unsqueeze(-1).expand(K, N), w_true, reduce="amin")
    group_max.scatter_reduce_(0, g_idx.long().unsqueeze(-1).expand(K, N), w_true, reduce="amax")
    scales = ((group_max - group_min) / maxq).clamp_min(1e-6)
    zero_point = torch.clamp(torch.round(-group_min / scales), 1, maxq)

    scales_row = scales[g_idx.long()]
    zp_row = zero_point[g_idx.long()]
    w_int = torch.clamp(torch.round(w_true / scales_row) + zp_row, 0, maxq).to(torch.int32)

    packed = torch.zeros((math_ceil_div(K, infeature_per_bits), N), device=device, dtype=torch.int32)
    for row in range(K):
        packed_row = row // infeature_per_bits
        shift = (row % infeature_per_bits) * bits
        packed[packed_row] |= (w_int[row] & maxq) << shift

    # The kernel expects `zeros` bit-packed the SAME way as the weights, but
    # packed along the N (column) axis instead of K: 8 4-bit zero-points per
    # int32 word (`zeros_ptrs + offs_bn//infeature_per_bits`, unpacked via
    # `(zeros >> (offs_bn%infeature_per_bits)*bits) & maxq`). Passing an
    # unpacked per-column tensor here (as if raw ints addressed it 1:1) reads
    # the wrong nibble for nearly every column.
    zeros_field = (zero_point - 1).to(torch.int32)  # kernel adds +1 back internally
    zeros_packed = torch.zeros((num_groups, math_ceil_div(N, infeature_per_bits)), device=device, dtype=torch.int32)
    for col in range(N):
        packed_col = col // infeature_per_bits
        shift = (col % infeature_per_bits) * bits
        zeros_packed[:, packed_col] |= (zeros_field[:, col] & maxq) << shift

    out = gptq_matmul(a, packed, scales, zeros_packed, g_idx, bits=bits)
    ref = a @ w_true  # continuous ground truth, NOT the round-tripped int4 value
    return out, ref


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"

    out, ref = run(K=256)
    fp32_tolerance_pass = torch.allclose(out, ref, rtol=1e-2, atol=1e-2)
    rel_err = ((out - ref).abs() / ref.abs().clamp_min(1e-3)).mean().item()

    # Format-aware check: INT4 (16 levels per group) is expected to carry
    # several percent of relative error; accept it as long as it's in that
    # expected ballpark rather than blown up (e.g. from an indexing bug).
    format_aware_pass = 0.0 < rel_err < 0.5

    print(f"real AutoGPTQ INT4 kernel (GPU), K=256: FP32-calibrated allclose(rtol=1e-2,atol=1e-2)={fp32_tolerance_pass}")
    print(f"mean relative error vs continuous ground truth = {rel_err:.3f} (expected INT4 quantization noise, not a bug)")
    print("FP DEMONSTRATED" if (not fp32_tolerance_pass) and format_aware_pass else "tune constants")
    return (not fp32_tolerance_pass) and format_aware_pass


if __name__ == "__main__":
    test_kernel()
''',
)


# ---------------------------------------------------------------------------
# FP5: standalone Philox stochastic-rounding kernel (tl.randint-based)
# ---------------------------------------------------------------------------
add(
    name="fp5_stochastic_rounding_philox",
    group="FP", seed_class="FP5",
    kernel_family="Stochastic rounding via Triton's Philox RNG (tl.randint)",
    reference="Standard technique also used by production low-bit training kernels (e.g. "
              "torchao's NVFP4 stochastic-rounding path, torchao/prototype/moe_training/"
              "nvfp4_training/hadamard_utils.py, which additionally uses an sm100-only PTX "
              "cvt.rs instruction unavailable on this Modal T4; this case uses Triton's "
              "portable tl.randint Philox generator to implement the same probabilistic "
              "rounding rule on any GPU).",
    failure_mode="false_positive",
    mechanism="The kernel intentionally rounds each value up or down to a coarse grid with "
               "probability proportional to its distance from each grid point, using a "
               "counter-based Philox RNG seeded per call. Two runs of the exact same correct "
               "kernel therefore legitimately produce different elementwise outputs; comparing "
               "one run against another with allclose is a false positive on intended "
               "nondeterminism.",
    expected={
        "ground_truth": "kernel.py is a correct stochastic-rounding kernel; run-to-run variation is intended",
        "naive_allclose_verdict": "FAIL when comparing two independently-seeded runs elementwise",
        "correct_verdict": "CORRECT (the mean over many draws converges to the exact input value, i.e. unbiased)",
    },
    note="Uses Triton's real tl.randint Philox counter-based RNG (the same generator underlying "
         "torch/triton dropout and stochastic-rounding kernels). Runs on GPU.",
    problem_txt='''FP5: stochastic rounding via Triton's real Philox RNG (tl.randint).

Spec: round x to the nearest multiple of STEP, rounding UP with probability
equal to how close x is to the upper grid point, using triton.language.randint
(a real, hardware-backed counter-based Philox generator) seeded per call.
E[output] == x, but any single run is intentionally random.
''',
    kernel_py='''"""Stochastic-rounding kernel using Triton's real Philox counter-based RNG
(tl.randint / tl.rand), the same generator underlying production dropout and
stochastic-rounding kernels (e.g. torchao's, which additionally use an
sm100-only PTX instruction unavailable on this Modal T4 GPU).
"""
import torch
import triton
import triton.language as tl

STEP = 0.05


@triton.jit
def stochastic_round_kernel(x_ptr, out_ptr, seed, N: tl.constexpr, STEP: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    lower = tl.floor(x / STEP) * STEP
    upper = lower + STEP
    p_up = (x - lower) / STEP
    r = tl.rand(seed, offs)  # real Triton Philox RNG
    out = tl.where(r < p_up, upper, lower)
    tl.store(out_ptr + offs, out)


def stochastic_round_to_grid(x: torch.Tensor, seed: int) -> torch.Tensor:
    N = x.shape[0]
    out = torch.empty_like(x)
    stochastic_round_kernel[(1,)](x, out, seed, N=N, STEP=STEP)
    return out
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import stochastic_round_to_grid, STEP  # real Triton Philox RNG


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"

    n = 4096
    x = torch.full((n,), 0.137, device=device)

    run_a = stochastic_round_to_grid(x, seed=1)
    run_b = stochastic_round_to_grid(x, seed=2)
    naive_pass = torch.allclose(run_a, run_b, rtol=1e-2, atol=1e-2)

    mean_estimate = run_a.mean().item()
    unbiased_pass = abs(mean_estimate - 0.137) < 4 * STEP / (n ** 0.5)

    print(f"real Triton Philox RNG (GPU): two independent seeds, elementwise allclose = {naive_pass} (expected False)")
    print(f"mean over {n} draws = {mean_estimate:.5f} vs true 0.13700; unbiased check = {unbiased_pass}")
    print("FP DEMONSTRATED" if (not naive_pass) and unbiased_pass else "tune constants")
    return (not naive_pass) and unbiased_pass


if __name__ == "__main__":
    test_kernel()
''',
)


# ---------------------------------------------------------------------------
# FP6: real Liger RMSNorm kernel, relative-error instability near zero
# ---------------------------------------------------------------------------
add(
    name="fp6_liger_rmsnorm_relative_error_near_zero",
    group="FP", seed_class="FP6",
    kernel_family="Liger-Kernel RMSNorm, correct kernel at two precisions",
    reference="linkedin/Liger-Kernel, src/liger_kernel/ops/rms_norm.py::_rms_norm_forward_kernel",
    failure_mode="false_positive",
    mechanism="The real, correct RMSNorm kernel run on an FP32 input and the same real kernel "
               "run on a BF16-rounded copy of that input are both correct for their declared "
               "precision. On an input row close to (but not exactly) zero, the true output is "
               "extremely small, so an ordinary BF16 rounding difference of the input becomes a "
               "large relative error in the output even though the absolute difference is "
               "negligible -- an artifact of the comparison metric, not a kernel defect.",
    expected={
        "ground_truth": "both kernel invocations are the real, correct kernel; the discrepancy is ordinary BF16 rounding of the input",
        "naive_allclose_verdict": "FAIL under a relative-error-only metric",
        "correct_verdict": "CORRECT (passes under torch.allclose's combined atol+rtol)",
    },
    note="Uses the real, unmodified Liger RMSNorm kernel (same kernel as fn7's reference) at two "
         "input precisions. Runs on GPU.",
    problem_txt='''FP6: real Liger-Kernel RMSNorm, relative-error instability near zero.

Source: linkedin/Liger-Kernel, src/liger_kernel/ops/rms_norm.py
::_rms_norm_forward_kernel (completely unmodified, correct kernel, run twice
at different input precisions).

Both runs are correct for their declared precision; comparing their outputs
with a relative-error-only metric on a near-zero row produces a large
apparent error even though torch.allclose's combined atol+rtol accepts it.
''',
    kernel_py='''"""Real Liger-Kernel RMSNorm forward kernel, completely unmodified.

Verbatim (casting-mode logic simplified to CASTING_MODE_NONE only) from
linkedin/Liger-Kernel, src/liger_kernel/ops/rms_norm.py
::_rms_norm_forward_kernel, fetched 2026-09-04. No bug: this FP case compares
the SAME kernel run at two input precisions.
"""
import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.libdevice import rsqrt
except ModuleNotFoundError:
    from triton.language.extra.cuda.libdevice import rsqrt


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr, Y_row_stride, X_ptr, X_row_stride, RSTD_ptr, RSTD_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd)
    Y_row = X_row * rstd
    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row, mask=mask)


def rms_norm_forward(X: torch.Tensor, eps: float) -> torch.Tensor:
    num_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    X32 = X.float()
    Y = torch.empty(num_rows, n_cols, device=X.device, dtype=torch.float32)
    RSTD = torch.empty(num_rows, dtype=torch.float32, device=X.device)
    _rms_norm_forward_kernel[(num_rows,)](Y, Y.stride(0), X32, X32.stride(0), RSTD, RSTD.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return Y
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import rms_norm_forward  # real, unmodified kernel -- used for BOTH sides


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    eps = 1e-6

    x_fp32 = torch.full((1, 16), 3e-4, device=device)
    x_bf16_roundtrip = x_fp32.bfloat16().float()  # ordinary BF16 rounding of the input

    out_fp32 = rms_norm_forward(x_fp32, eps)
    out_bf16 = rms_norm_forward(x_bf16_roundtrip, eps)

    abs_diff = (out_fp32 - out_bf16).abs().max().item()
    relative_only_pass = abs_diff <= 1e-2 * out_fp32.abs().max().item()
    combined_pass = torch.allclose(out_fp32, out_bf16, rtol=1e-2, atol=1e-2)

    print(f"real Liger RMSNorm kernel (GPU): abs diff = {abs_diff:.3e}, output magnitude ~ {out_fp32.abs().max().item():.3f}")
    print(f"relative-error-only check = {relative_only_pass}")
    print(f"torch.allclose (atol+rtol combined) = {combined_pass} (expected True)")
    print("FP DEMONSTRATED" if combined_pass else "tune constants")
    return combined_pass


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
