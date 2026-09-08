"""Build the answer-free evaluation copy of the FN/FP Triton benchmark.

The authored cases in benchmark_fn_fp/triton/ double as documentation: their
problem.txt and kernel.py spell out what was changed and whether the kernel is
correct. That is fine for a human reader and fatal for an evaluation -- any
verifier reading them is just copying the answer.

This script emits benchmark_fn_fp/eval_cases/<name>/ containing ONLY what a
verifier should legitimately see:

    problem.txt  - the operator contract, rewritten neutrally (below). States
                   what must hold, and what the spec deliberately leaves
                   unspecified, for BOTH groups symmetrically. Never says
                   whether kernel.py conforms.
    kernel.py    - byte-identical code, with the leaking module docstring and
                   `# BUG: ...` comments replaced by a neutral header.
    meta.json    - name/status only. No group, no seed_class, no mechanism,
                   no expected verdict.

test.py is deliberately NOT copied: it contains the reference implementation
and prints the demonstration outcome.

The answer key stays in benchmark_fn_fp/triton/<name>/meta.json, which the
evaluation harness reads separately for scoring.

Run from repo root:  python benchmark_fn_fp/generation/generators/sanitize_for_eval.py
"""
from __future__ import annotations

import json
import os
import re

SRC_DIR = os.environ.get("KV_SRC_DIR", "benchmark_fn_fp/triton")
OUT_DIR = os.environ.get("KV_EVAL_DIR", "benchmark_fn_fp/eval_cases")


# Neutral contract statements. Each says what the operation must satisfy and
# what is left open -- never whether this particular kernel.py satisfies it.
PROBLEM_STATEMENTS: dict[str, str] = {
    "fn1_nsa_bitonic_topk_tie_dilution": '''
Operation: top-k selection over candidate block scores, implemented with the
bitonic sort network used by Native Sparse Attention's selection branch
(fla-org/native-sparse-attention, native_sparse_attention/ops/utils.py).

`sorted_topk_indices(scores, k)` must return the indices of the k
highest-scoring candidates, in descending score order.

Contract on ties: when two candidates have exactly equal scores, the
LOWER index must be the one kept.

Downstream, the selected indices gather value vectors that are averaged
together, so the selection feeds a continuous output.
''',
    "fn1_nsa_bitonic_gate_tie_pooled": '''
Operation: top-1 routing (a gate) over per-expert scores, implemented with the
bitonic sort network used by Native Sparse Attention's selection branch
(fla-org/native-sparse-attention, native_sparse_attention/ops/utils.py).

`sorted_topk_indices(scores, k=1)` must return, for each token, the index of
the highest-scoring expert.

Contract on ties: when two experts have exactly equal scores, the LOWER
index must be the one kept.

Each token's chosen expert contributes its output vector; the batch is then
mean-pooled into a single continuous representation.
''',
    "fn2_vllm_rejection_sampler_missing_clip": '''
Operation: the recovered-token sampler used by speculative decoding
(vllm-project/vllm, vllm/v1/sample/rejection_sampler.py::sample_recovered_tokens_kernel).

After a draft token is rejected, a replacement token must be sampled from the
residual distribution

    p_residual(v)  proportional to  max(target_prob[v] - draft_prob[v], 0)

i.e. a token the draft over-proposed (target_prob < draft_prob) carries zero
residual probability and must never be returned as the recovered token.

The kernel samples from that distribution with the exponential-race
(Gumbel-max) trick: given q[v] ~ Exponential(1) supplied as inv_q = 1/q, it
returns argmax_v( p_residual(v) * inv_q[v] ).
''',
    "fn3_gptq_dequant_group_div_coverage": '''
Operation: grouped INT4 dequantization fused with a matrix multiply
(AutoGPTQ/AutoGPTQ, auto_gptq/nn_modules/triton_utils/kernels.py::quant_matmul_248_kernel).

Weights are stored as 4-bit values packed into int32 words. Columns along the
reduction dimension K are split into groups of `group_size`; each group has
its own scale and zero-point, and a per-column table `g_idx` maps each column
to its group.

Contract: every column must be dequantized with the scale of the group it
actually belongs to, for any K -- including a K that is not an exact multiple
of group_size, where the trailing partial group is still a group of its own.

The kernel computes  C = A @ dequant(B),  with dequant(B)[k,n] =
(unpacked_q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n].
''',
    "fn4_gqa_head_mapping_homogeneous": '''
Operation: the query-head to KV-head mapping of Grouped Query Attention, in
the style of the index arithmetic in vllm-project/vllm,
vllm/v1/attention/ops/triton_unified_attention.py.

With num_q_heads query heads and num_kv_heads key/value heads
(n_rep = num_q_heads // num_kv_heads), GQA groups CONSECUTIVE query heads onto
the same KV head: query heads 0..n_rep-1 read KV head 0, query heads
n_rep..2*n_rep-1 read KV head 1, and so on.

`gqa_gather(kv, num_q_heads)` must return, for each query head, the KV row
that query head is supposed to read.
''',
    "fn5_sglang_int8_quant_stale_scale": '''
Operation: per-token INT8 activation quantization
(sgl-project/sglang, python/sglang/kernels/ops/quantization/int8_kernel.py::_per_token_quant_int8).

For each row of the input, the kernel must quantize to int8 and return the
dequantized reconstruction.

Contract: the scale is per-row and derived from THAT row's own magnitude:

    absmax = max(max(|x_row|), 1e-10)
    scale  = absmax / 127
    q      = clamp(round(x_row / scale), -127, 127)
    out    = q * scale

so no value in the row is clipped by its own scale.
''',
    "fn6_mamba_state_passing_lowbit_drift": '''
Operation: the chunked state-passing recurrence of Mamba2/SSD
(state-spaces/mamba, mamba_ssm/ops/triton/ssd_state_passing.py::_state_passing_fwd_kernel).

Given per-chunk local states `new_states[c]` and per-chunk cumulative
log-decays `dA_cs[c]`, the kernel must carry the recurrent state across chunks
exactly:

    state <- exp(dA_cs[c]) * state + new_states[c]        for c = 0..nchunks-1

starting from state = 0, and return the final state after the last chunk.

The recurrence is exact: it introduces no approximation of its own beyond
ordinary floating-point arithmetic, at any sequence length.
''',
    "fn7_liger_rmsnorm_eps_placement": '''
Operation: RMSNorm forward
(linkedin/Liger-Kernel, src/liger_kernel/ops/rms_norm.py::_rms_norm_forward_kernel).

For each row x:

    mean_square = sum(x * x) / n_cols
    rstd        = 1 / sqrt(mean_square + eps)
    y           = x * rstd

The epsilon is the standard RMSNorm stabilizer: it must keep the normalizer
finite as the row norm approaches zero, so that a row of (near-)zero input
does not blow the output up.
''',
    "fp1_vllm_topp_tie_epsilon": '''
Operation: the near-pivot tie-counting primitive used by top-k/top-p sampling
(vllm-project/vllm, vllm/v1/sample/ops/topk_topp_triton.py::_update_min_larger_stats).

`count_tied_at_boundary(scores, pivot)` must return how many of the values
strictly above `pivot` are tied with the smallest of them.

Because scores are floating point, "tied" is decided with a tolerance rather
than exact equality. The specification requires only that values equal at the
boundary be counted together; it does not fix the numeric value of that
tolerance.

Downstream, the returned count determines how a remaining unit of selection
budget is split among the boundary candidates (1 / count each).
''',
    "fp2_flashattn_rope_neox_vs_gptj": '''
Operation: rotary position embeddings (RoPE)
(Dao-AILab/flash-attention, flash_attn/ops/triton/rotary.py::rotary_kernel).

`apply_rotary(x, cos, sin, interleaved)` must rotate each pair of feature
dimensions of x by the position-dependent angle encoded in cos/sin, so that
the dot product of a rotated query and a rotated key depends on their relative
position.

Two pairings are in use in the field and the kernel implements both behind the
`interleaved` flag:
  - interleaved=False: dimension i is paired with dimension i + headdim/2
  - interleaved=True:  dimension 2i is paired with dimension 2i+1

The specification does not mandate which pairing a model uses; it requires
only that a given model apply the same one consistently to both q and k.
''',
    "fp3_mamba_state_passing_chunk_granularity": '''
Operation: the chunked state-passing recurrence of Mamba2/SSD
(state-spaces/mamba, mamba_ssm/ops/triton/ssd_state_passing.py::_state_passing_fwd_kernel).

Given per-chunk local states `new_states[c]` and per-chunk cumulative
log-decays `dA_cs[c]`:

    state <- exp(dA_cs[c]) * state + new_states[c]        for c = 0..nchunks-1

starting from state = 0, returning the final state.

The chunk boundaries are a partitioning choice, not part of the mathematical
definition: subdividing one chunk into several sub-chunks whose log-decays sum
to the original, with the local-state contribution placed on the last
sub-chunk, describes the same recurrence.
''',
    "fp4_gptq_int4_quant_tolerance": '''
Operation: grouped INT4 dequantization fused with a matrix multiply
(AutoGPTQ/AutoGPTQ, auto_gptq/nn_modules/triton_utils/kernels.py::quant_matmul_248_kernel).

The kernel computes  C = A @ dequant(B),  where B holds 4-bit weights packed
into int32 words with a per-(group, column) scale and zero-point, and

    dequant(B)[k,n] = (unpacked_q[k,n] - zero[g_idx[k],n]) * scale[g_idx[k],n]

The declared numeric format of the weights is 4-bit: the weight values it
consumes were produced by quantizing continuous weights onto 16 levels per
group, and the kernel's job is to evaluate the matmul faithfully in that
declared format.
''',
    "fp5_stochastic_rounding_philox": '''
Operation: stochastic rounding of a tensor onto a coarse grid, using Triton's
counter-based Philox RNG (tl.rand), as used by low-precision training kernels
when casting high-precision accumulators down to a storage format.

`stochastic_round_to_grid(x, seed)` must round each element to one of the two
neighbouring multiples of STEP, choosing the upper one with probability equal
to the element's fractional distance to it:

    lower = floor(x / STEP) * STEP
    P(out = lower + STEP) = (x - lower) / STEP,   otherwise out = lower

so that E[out] == x. The randomness is drawn from the supplied seed.
''',
    "fp6_liger_rmsnorm_relative_error_near_zero": '''
Operation: RMSNorm forward
(linkedin/Liger-Kernel, src/liger_kernel/ops/rms_norm.py::_rms_norm_forward_kernel).

For each row x:

    mean_square = sum(x * x) / n_cols
    rstd        = 1 / sqrt(mean_square + eps)
    y           = x * rstd

computed with float32 accumulation. The kernel is expected to be run on inputs
that may arrive at different storage precisions (e.g. an fp32 tensor, or the
same tensor after a bf16 round-trip), and on rows whose magnitude is close to
zero.
''',
}


_LEAK_COMMENT = re.compile(r"\s*#\s*(BUG|NOTE: the real|real kernel|real mapping|a different, still-reasonable).*$",
                           re.IGNORECASE)


def sanitize_kernel(code: str, name: str) -> str:
    """Strip the leaking module docstring and inline verdict comments.

    Only comments/docstrings are touched; every executable line is preserved
    byte-for-byte so the sanitized kernel behaves identically.
    """
    lines = code.splitlines()

    # Drop the module docstring (it names the mutation / says "no bug").
    if lines and lines[0].lstrip().startswith('"""'):
        end = 0
        if lines[0].count('"""') >= 2:
            end = 0
        else:
            for i in range(1, len(lines)):
                if '"""' in lines[i]:
                    end = i
                    break
        lines = lines[end + 1:]
        while lines and not lines[0].strip():
            lines.pop(0)

    cleaned = []
    for line in lines:
        stripped = _LEAK_COMMENT.sub("", line)
        # A comment line that existed only to flag the mutation goes entirely.
        if stripped.strip() == "" and line.strip().startswith("#"):
            continue
        cleaned.append(stripped.rstrip())

    header = f'"""Triton kernel under test: {name}."""\n'
    return header + "\n".join(cleaned).strip() + "\n"


def main() -> None:
    names = sorted(
        d for d in os.listdir(SRC_DIR)
        if os.path.isdir(os.path.join(SRC_DIR, d))
        and os.path.exists(os.path.join(SRC_DIR, d, "meta.json"))
    )
    missing = [n for n in names if n not in PROBLEM_STATEMENTS]
    if missing:
        raise SystemExit(f"no neutral problem statement written for: {missing}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for name in names:
        src = os.path.join(SRC_DIR, name)
        dst = os.path.join(OUT_DIR, name)
        os.makedirs(dst, exist_ok=True)

        neutral_problem = PROBLEM_STATEMENTS[name].strip() + "\n"
        with open(os.path.join(src, "kernel.py")) as f:
            kernel_code = f.read()
        clean_kernel = sanitize_kernel(kernel_code, name)

        # The benchmark itself must be answer-free: rewrite the canonical case
        # in place. The full explanation of what each case tests stays in the
        # canonical meta.json, which the scorer reads and no verifier sees.
        with open(os.path.join(src, "problem.txt"), "w") as f:
            f.write(neutral_problem)
        with open(os.path.join(src, "kernel.py"), "w") as f:
            f.write(clean_kernel)

        # The evaluation copy additionally drops test.py (it carries the
        # reference implementation and the demonstration outcome) and every
        # answer-bearing meta.json field.
        with open(os.path.join(dst, "problem.txt"), "w") as f:
            f.write(neutral_problem)
        with open(os.path.join(dst, "kernel.py"), "w") as f:
            f.write(clean_kernel)
        with open(os.path.join(dst, "meta.json"), "w") as f:
            json.dump({"name": name, "status": "under_test", "passed": None}, f, indent=2)
            f.write("\n")

    print(f"rewrote {len(names)} canonical cases answer-free in {os.path.abspath(SRC_DIR)}")
    print(f"wrote {len(names)} verifier-input cases to {os.path.abspath(OUT_DIR)}")

    # Fail loudly if any verdict language survived, in either location.
    banned = re.compile(r"\b(bug|buggy|violat|defect|unmodified|injected|mutat|not a bug|correct kernel|both are valid)\b",
                        re.IGNORECASE)
    leaks = []
    for root in (SRC_DIR, OUT_DIR):
        for name in names:
            for fname in ("problem.txt", "kernel.py"):
                path = os.path.join(root, name, fname)
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    for lineno, line in enumerate(f, 1):
                        if banned.search(line):
                            leaks.append(f"{path}:{lineno}: {line.strip()}")
    if leaks:
        print("\nLEAK CHECK FAILED:")
        for leak in leaks:
            print("  " + leak)
        raise SystemExit(1)
    print("leak check passed: no verdict language in any problem.txt or kernel.py")


if __name__ == "__main__":
    main()
