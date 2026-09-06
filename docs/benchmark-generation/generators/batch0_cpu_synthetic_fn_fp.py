"""Generate the first batch of FN/FP adversarial-allclose benchmark cases.

Writes benchmark_fn_fp/<name>/{problem.txt,kernel.py,test.py,meta.json} for
each case -- kept separate from dataset/ (the kernelagent/_advprec_* corpus).
All cases are self-contained, CPU-only PyTorch (no CUDA/Triton needed).

Run from the repo root: `python docs/benchmark-generation/generators/batch1_fn_fp.py`
Point kv-agentic-run at it with `--dataset-dir benchmark_fn_fp`.
"""
import json
import os

# Run from the repo root; override with KV_DATASET_DIR to write elsewhere.
DATASET_DIR = os.environ.get("KV_DATASET_DIR", "benchmark_fn_fp")

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
# FN1a: top-k tie-break bug hidden by softmax dilution
# ---------------------------------------------------------------------------
add(
    name="fn1_topk_tieblock_dilution",
    group="FN", seed_class="FN1",
    kernel_family="NSA-style top-k block selection for sparse attention",
    reference="https://arxiv.org/abs/2502.11089 (Native Sparse Attention, selection branch)",
    failure_mode="false_negative",
    mechanism="Manual top-k uses '>=' instead of '>' when updating the running best, so a "
               "later, higher-index block silently wins an exact score tie instead of the "
               "lower-index block the contract requires. The swapped block's contribution is "
               "then averaged with k-1 other correctly selected blocks, diluting the error "
               "below the default allclose tolerance.",
    expected={
        "ground_truth": "kernel.py violates the stated tie-break contract (lower index must win)",
        "naive_allclose_verdict": "PASS at rtol=1e-2/atol=1e-2 (bug is hidden)",
        "correct_verdict": "BUGGY",
    },
    note="test.py builds an exact tie at the k-th boundary between index a<b, gives each "
         "candidate a distinct value vector, and shows the k=8 averaged output stays inside "
         "tolerance even though one of the 8 selected indices is contractually wrong.",
    problem_txt='''"""FN1: deterministic top-k tie-breaking contract.

Spec: `select_topk(scores, k)` must return the indices of the k highest-scoring
candidates. When two candidates have exactly equal scores, the LOWER index must
win (this is a stated part of the contract, not left ambiguous).

Reference implementation (ground truth):
''' + '''
import torch

def select_topk_reference(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Stable: on an exact tie, the lower index wins because argsort is
    stable and we sort ascending on -scores (ties keep original order)."""
    order = torch.argsort(-scores, dim=-1, stable=True)
    return order[..., :k]
''',
    kernel_py='''import torch


def select_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Candidate implementation under test.

    BUG: uses `cur >= best_val` instead of `cur > best_val`, so on an exact
    tie the later (higher-index) candidate overwrites the earlier one.
    """
    scores = scores.clone()
    batch_shape = scores.shape[:-1]
    n = scores.shape[-1]
    out = torch.zeros(*batch_shape, k, dtype=torch.long)
    for step in range(k):
        best_idx = torch.zeros(batch_shape, dtype=torch.long)
        best_val = torch.full(batch_shape, float("-inf"))
        for i in range(n):
            cur = scores[..., i]
            take = cur >= best_val  # BUG: should be strictly '>'
            best_idx = torch.where(take, torch.full_like(best_idx, i), best_idx)
            best_val = torch.where(take, cur, best_val)
        out[..., step] = best_idx
        scores.scatter_(-1, best_idx.unsqueeze(-1), float("-inf"))
    return out
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import select_topk  # candidate (buggy)


def select_topk_reference(scores: torch.Tensor, k: int) -> torch.Tensor:
    order = torch.argsort(-scores, dim=-1, stable=True)
    return order[..., :k]


def run_downstream(scores, values, k, select_fn):
    idx = select_fn(scores, k)
    gathered = torch.gather(
        values.unsqueeze(0).expand(scores.shape[0], -1, -1),
        1,
        idx.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
    )
    return gathered.mean(dim=1)  # averaged "attention-like" output over the k selected blocks


def test_kernel():
    torch.manual_seed(0)
    n, k, d = 32, 8, 16
    scores = torch.randn(1, n)
    # Force an exact tie between index 7 (correct winner, lower index) and
    # index 20 (buggy winner, higher index) right at the k-th boundary.
    sorted_vals, _ = torch.sort(scores[0], descending=True)
    tie_value = sorted_vals[k - 1].item()
    scores[0, 7] = tie_value
    scores[0, 20] = tie_value

    values = torch.randn(n, d) * 0.3
    # Keep the two tied candidates' content close (as if both blocks summarize
    # similar nearby context) so the swap is diluted below tolerance once
    # averaged with the k-1 other correctly selected blocks.
    values[20] = values[7] + torch.randn(d) * 0.03

    ref_idx = select_topk_reference(scores, k)
    cand_idx = select_topk(scores, k)
    assert 7 in ref_idx[0].tolist() and 20 not in ref_idx[0].tolist(), "reference should pick idx 7"
    assert 20 in cand_idx[0].tolist() and 7 not in cand_idx[0].tolist(), "bug should surface as idx 20 winning"

    ref_out = run_downstream(scores, values, k, select_topk_reference)
    cand_out = run_downstream(scores, values, k, select_topk)

    naive_pass = torch.allclose(cand_out, ref_out, rtol=1e-2, atol=1e-2)
    max_abs = (cand_out - ref_out).abs().max().item()
    print(f"selected indices differ (contract violation): ref has 7, candidate has 20 instead")
    print(f"downstream max abs diff = {max_abs:.6f}, naive allclose(rtol=1e-2,atol=1e-2) = {naive_pass}")
    print("FN DEMONSTRATED" if naive_pass else "tolerance too tight for this seed; tune values")
    return naive_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FN1b: MoE top-1 gating tie hidden by batch-mean pooling
# ---------------------------------------------------------------------------
add(
    name="fn1_moe_gate_argmax_pooled",
    group="FN", seed_class="FN1",
    kernel_family="MoE top-1 gating (argmax router)",
    reference="Switch Transformer / MoE top-1 routing style; extension of FN1's tie-break contract",
    failure_mode="false_negative",
    mechanism="Router contract requires the lower-index expert to win an exact logit tie. The "
               "buggy argmax uses a last-max-wins scan. Only one token (out of many) in the "
               "batch hits the tie, and the verifier only compares the batch-mean pooled "
               "output, so the single wrong routing decision is averaged away.",
    expected={
        "ground_truth": "kernel.py violates the tie-break contract on one token",
        "naive_allclose_verdict": "PASS on the pooled batch mean",
        "correct_verdict": "BUGGY",
    },
    note="test.py builds a batch of 256 tokens, ties exactly one token's top-2 expert logits, "
         "and compares mean-pooled gated output across the whole batch.",
    problem_txt='''"""FN1 (extension): MoE top-1 router tie-break contract.

Spec: `route(logits)` must return, for each token, the index of the highest
scoring expert; on an exact tie the LOWER index wins.

Reference implementation (ground truth):
''' + '''
import torch

def route_reference(logits: torch.Tensor) -> torch.Tensor:
    return torch.argsort(-logits, dim=-1, stable=True)[..., 0]
''',
    kernel_py='''import torch


def route(logits: torch.Tensor) -> torch.Tensor:
    """Candidate router. BUG: last-max-wins on ties (should be first-max-wins)."""
    n_tokens, n_experts = logits.shape
    out = torch.zeros(n_tokens, dtype=torch.long)
    for t in range(n_tokens):
        best_idx = 0
        best_val = float("-inf")
        for e in range(n_experts):
            if logits[t, e] >= best_val:  # BUG: should be strictly '>'
                best_idx = e
                best_val = logits[t, e].item()
        out[t] = best_idx
    return out
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import route  # candidate (buggy)


def route_reference(logits: torch.Tensor) -> torch.Tensor:
    return torch.argsort(-logits, dim=-1, stable=True)[..., 0]


def test_kernel():
    torch.manual_seed(0)
    n_tokens, n_experts, d = 256, 4, 8
    logits = torch.randn(n_tokens, n_experts)
    # tie exactly one token's top-2 experts (index 0 should win, index 1 must not)
    logits[10, 0] = 5.0
    logits[10, 1] = 5.0

    expert_weight = torch.randn(n_experts, d)  # each expert has a distinct output vector

    ref_idx = route_reference(logits)
    cand_idx = route(logits)
    assert ref_idx[10].item() == 0
    assert cand_idx[10].item() == 1, "bug should surface as expert 1 winning token 10's tie"

    ref_out = expert_weight[ref_idx]
    cand_out = expert_weight[cand_idx]

    pooled_ref = ref_out.mean(dim=0)
    pooled_cand = cand_out.mean(dim=0)

    naive_pass = torch.allclose(pooled_cand, pooled_ref, rtol=1e-2, atol=1e-2)
    per_token_pass = torch.allclose(cand_out, ref_out, rtol=1e-2, atol=1e-2)
    print(f"pooled diff = {(pooled_cand - pooled_ref).abs().max().item():.6f}, naive allclose on pooled mean = {naive_pass}")
    print(f"per-token allclose (would catch it) = {per_token_pass}")
    print("FN DEMONSTRATED" if naive_pass and not per_token_pass else "tune constants")
    return naive_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FN2: speculative decoding rejection-sampling missing clamp
# ---------------------------------------------------------------------------
add(
    name="fn2_speculative_reject_clip",
    group="FN", seed_class="FN2",
    kernel_family="vLLM-style speculative decoding rejection sampler",
    reference="https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md",
    failure_mode="false_negative",
    mechanism="The corrected distribution after a draft-token rejection must clamp "
               "(target_prob - draft_prob) to zero before renormalizing. The buggy kernel "
               "skips the clamp. On mild, near-identical draft/target distributions the "
               "negative mass is negligible and both formulas agree; the bug only shows up "
               "once draft and target diverge enough to produce sizeable negative terms.",
    expected={
        "ground_truth": "kernel.py omits max(0, ...) before renormalizing",
        "naive_allclose_verdict": "PASS on mild (typical) distributions, FAIL only on adversarial ones",
        "correct_verdict": "BUGGY",
    },
    note="test.py runs the same two implementations on a 'mild' distribution pair (draft"
         "≈target) and an 'adversarial' pair (draft far from target), showing the bug is"
         " invisible under typical test data.",
    problem_txt='''"""FN2: speculative-decoding rejection correction distribution.

Spec: given target_prob and draft_prob over a vocabulary, the corrected
distribution used after rejecting a draft token is

    p_correct = clamp(target_prob - draft_prob, min=0)
    p_correct = p_correct / p_correct.sum()

Reference implementation (ground truth):
''' + '''
import torch

def corrected_distribution_reference(target_prob, draft_prob):
    diff = torch.clamp(target_prob - draft_prob, min=0.0)
    return diff / diff.sum(dim=-1, keepdim=True)
''',
    kernel_py='''import torch


def corrected_distribution(target_prob: torch.Tensor, draft_prob: torch.Tensor) -> torch.Tensor:
    """Candidate implementation. BUG: omits the clamp(..., min=0) step."""
    diff = target_prob - draft_prob  # BUG: missing torch.clamp(diff, min=0.0)
    return diff / diff.sum(dim=-1, keepdim=True)
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import corrected_distribution  # candidate (buggy)


def corrected_distribution_reference(target_prob, draft_prob):
    diff = torch.clamp(target_prob - draft_prob, min=0.0)
    return diff / diff.sum(dim=-1, keepdim=True)


def test_kernel():
    vocab = 32

    # target_prob/draft_prob are evaluated over a restricted candidate set
    # here (a realistic simplification: rejection sampling is often applied
    # over a truncated top-k/top-p support, which need not sum to 1 on its
    # own). Mild case: draft is pointwise <= target everywhere -> diff has NO
    # negative entries, so clamp() is a no-op and the two formulas coincide
    # exactly. This is the common case ordinary tests exercise (draft rarely
    # over-proposes relative to target).
    target_mild = torch.zeros(vocab)
    target_mild[0], target_mild[1] = 0.5, 0.3
    draft_mild = torch.zeros(vocab)
    draft_mild[0], draft_mild[1] = 0.499, 0.299  # <= target everywhere
    ref_mild = corrected_distribution_reference(target_mild.unsqueeze(0), draft_mild.unsqueeze(0))
    cand_mild = corrected_distribution(target_mild.unsqueeze(0), draft_mild.unsqueeze(0))
    mild_pass = torch.allclose(cand_mild, ref_mild, rtol=1e-2, atol=1e-2)

    # Adversarial case: draft over-proposes one token beyond target (a real
    # rejection). Now diff has a genuine negative entry, so the raw
    # (unclamped) sum used by the bug is smaller than the correct clamped
    # sum -- and can even include negative "probabilities".
    target_adv = torch.zeros(vocab)
    target_adv[0], target_adv[1], target_adv[2] = 0.9, 0.05, 0.05
    draft_adv = torch.zeros(vocab)
    draft_adv[0], draft_adv[1] = 0.85, 0.10  # over-proposes token 1 beyond target
    ref_adv = corrected_distribution_reference(target_adv.unsqueeze(0), draft_adv.unsqueeze(0))
    cand_adv = corrected_distribution(target_adv.unsqueeze(0), draft_adv.unsqueeze(0))
    adv_pass = torch.allclose(cand_adv, ref_adv, rtol=1e-2, atol=1e-2)

    print(f"mild-distribution allclose = {mild_pass} (bug hidden on typical test data)")
    print(f"adversarial-distribution allclose = {adv_pass} (bug exposed only when draft/target diverge)")
    print("FN DEMONSTRATED" if mild_pass and not adv_pass else "tune constants")
    return mild_pass and not adv_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FN3a: AutoGPTQ-style group-count floor vs ceil division
# ---------------------------------------------------------------------------
add(
    name="fn3_gptq_group_div_coverage",
    group="FN", seed_class="FN3",
    kernel_family="AutoGPTQ-style grouped INT4 dequantization",
    reference="https://github.com/AutoGPTQ/AutoGPTQ",
    failure_mode="false_negative",
    mechanism="num_groups must be ceil(hidden_size/group_size) so the trailing partial group "
               "gets its own scale. The buggy kernel uses floor division, so on a hidden_size "
               "divisible by group_size the two formulas coincide and outputs match bit for "
               "bit; only an irregular hidden_size exposes the missing tail group.",
    expected={
        "ground_truth": "kernel.py uses floor division for num_groups instead of ceil",
        "naive_allclose_verdict": "PASS on divisible hidden_size (4096), FAIL on irregular (300)",
        "correct_verdict": "BUGGY",
    },
    note="test.py dequantizes with hidden_size=4096 (divisible: bit-exact match) and "
         "hidden_size=300 (irregular: tail 44 columns reuse the wrong group scale).",
    problem_txt='''"""FN3: grouped dequantization group-count contract.

Spec: given hidden_size and group_size, columns are split into
ceil(hidden_size / group_size) groups; each group has its own scale, and the
trailing partial group (if any) still gets its own scale.

Reference implementation (ground truth):
''' + '''
import math
import torch

def dequantize_reference(q: torch.Tensor, group_size: int, scales: torch.Tensor) -> torch.Tensor:
    hidden_size = q.shape[-1]
    num_groups = math.ceil(hidden_size / group_size)
    assert scales.shape[-1] == num_groups
    g_idx = torch.clamp(torch.arange(hidden_size) // group_size, max=num_groups - 1)
    return q * scales[g_idx]
''',
    kernel_py='''import torch


def dequantize(q: torch.Tensor, group_size: int, scales: torch.Tensor) -> torch.Tensor:
    """Candidate implementation. BUG: floor division for num_groups drops the
    trailing partial group, so its columns reuse the last full group's scale
    instead of getting their own."""
    hidden_size = q.shape[-1]
    num_groups_bug = hidden_size // group_size  # BUG: should be ceil
    g_idx = torch.clamp(torch.arange(hidden_size) // group_size, max=num_groups_bug - 1)
    return q * scales[g_idx]
''',
    test_py='''import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import dequantize  # candidate (buggy)


def dequantize_reference(q: torch.Tensor, group_size: int, scales: torch.Tensor) -> torch.Tensor:
    hidden_size = q.shape[-1]
    num_groups = math.ceil(hidden_size / group_size)
    assert scales.shape[-1] == num_groups
    g_idx = torch.clamp(torch.arange(hidden_size) // group_size, max=num_groups - 1)
    return q * scales[g_idx]


def run(hidden_size, group_size):
    torch.manual_seed(0)
    num_groups = math.ceil(hidden_size / group_size)
    q = torch.randint(-8, 8, (4, hidden_size)).float()
    scales = torch.linspace(1.0, 1.0 + 0.5 * (num_groups - 1), num_groups)  # distinct per-group scale
    ref = dequantize_reference(q, group_size, scales)
    cand = dequantize(q, group_size, scales[: hidden_size // group_size] if hidden_size % group_size else scales)
    # candidate only has weights for its own (possibly smaller) number of groups
    return torch.allclose(cand, ref, rtol=1e-2, atol=1e-2), (cand - ref).abs().max().item()


def test_kernel():
    divisible_pass, divisible_diff = run(hidden_size=4096, group_size=128)
    irregular_pass, irregular_diff = run(hidden_size=300, group_size=128)
    print(f"divisible hidden_size=4096: allclose={divisible_pass}, max_diff={divisible_diff:.6f} (bug invisible)")
    print(f"irregular hidden_size=300: allclose={irregular_pass}, max_diff={irregular_diff:.6f} (bug exposed)")
    print("FN DEMONSTRATED" if divisible_pass and not irregular_pass else "tune constants")
    return divisible_pass and not irregular_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FN4: GQA head-mapping bug hidden by homogeneous KV data
# ---------------------------------------------------------------------------
add(
    name="fn4_gqa_head_mapping_homogeneous",
    group="FN", seed_class="FN4",
    kernel_family="Grouped Query Attention (GQA) query-to-KV head mapping",
    reference="https://arxiv.org/abs/2305.13245",
    failure_mode="false_negative",
    mechanism="The mapping must be kv_head = query_head // n_rep. The buggy kernel uses "
               "query_head % num_kv_heads instead, producing a completely different grouping. "
               "If every KV head is filled with the same (or highly similar) content, any "
               "mapping -- right or wrong -- reads indistinguishable data, so the outputs "
               "coincide even though the routing formula is objectively wrong.",
    expected={
        "ground_truth": "kernel.py maps query heads to KV heads with the wrong formula",
        "naive_allclose_verdict": "PASS when KV heads carry homogeneous data, FAIL when KV heads carry distinguishable signatures",
        "correct_verdict": "BUGGY",
    },
    note="test.py runs the same buggy mapping against (a) KV heads broadcast from one shared "
         "vector and (b) KV heads with a distinct per-head signature.",
    problem_txt='''"""FN4: GQA query-to-KV head mapping contract.

Spec: with num_q_heads query heads and num_kv_heads key/value heads
(n_rep = num_q_heads // num_kv_heads), each query head must read from
kv_head = query_head // n_rep.

Reference implementation (ground truth):
''' + '''
import torch

def gather_kv_reference(kv: torch.Tensor, num_q_heads: int, n_rep: int) -> torch.Tensor:
    # kv: [num_kv_heads, dim]
    mapping = torch.arange(num_q_heads) // n_rep
    return kv[mapping]
''',
    kernel_py='''import torch


def gather_kv(kv: torch.Tensor, num_q_heads: int, n_rep: int) -> torch.Tensor:
    """Candidate implementation. BUG: uses modulo instead of floor division."""
    num_kv_heads = kv.shape[0]
    mapping = torch.arange(num_q_heads) % num_kv_heads  # BUG: should be // n_rep
    return kv[mapping]
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import gather_kv  # candidate (buggy)


def gather_kv_reference(kv: torch.Tensor, num_q_heads: int, n_rep: int) -> torch.Tensor:
    mapping = torch.arange(num_q_heads) // n_rep
    return kv[mapping]


def test_kernel():
    torch.manual_seed(0)
    num_q_heads, num_kv_heads, dim = 8, 2, 4
    n_rep = num_q_heads // num_kv_heads

    ref_mapping = torch.arange(num_q_heads) // n_rep
    cand_mapping = torch.arange(num_q_heads) % num_kv_heads
    assert not torch.equal(ref_mapping, cand_mapping), "mappings must actually differ"

    # (a) homogeneous KV data: every head is the same base vector -> bug invisible
    base = torch.randn(1, dim)
    kv_homogeneous = base.expand(num_kv_heads, dim).clone()
    ref_out_h = gather_kv_reference(kv_homogeneous, num_q_heads, n_rep)
    cand_out_h = gather_kv(kv_homogeneous, num_q_heads, n_rep)
    homogeneous_pass = torch.allclose(cand_out_h, ref_out_h, rtol=1e-2, atol=1e-2)

    # (b) distinguishable KV data: each head gets its own signature -> bug exposed
    kv_distinct = torch.stack([torch.full((dim,), float(i * 100)) for i in range(num_kv_heads)])
    ref_out_d = gather_kv_reference(kv_distinct, num_q_heads, n_rep)
    cand_out_d = gather_kv(kv_distinct, num_q_heads, n_rep)
    distinct_pass = torch.allclose(cand_out_d, ref_out_d, rtol=1e-2, atol=1e-2)

    print(f"homogeneous KV data: allclose={homogeneous_pass} (bug hidden)")
    print(f"distinguishable KV data: allclose={distinct_pass} (bug exposed)")
    print("FN DEMONSTRATED" if homogeneous_pass and not distinct_pass else "tune constants")
    return homogeneous_pass and not distinct_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FN5: INT8 EMA-scale outlier
# ---------------------------------------------------------------------------
add(
    name="fn5_int8_ema_scale_outlier",
    group="FN", seed_class="FN5",
    kernel_family="INT8 activation quantization scale computation",
    reference="LLM.int8() (https://arxiv.org/abs/2208.07339) / SmoothQuant (https://arxiv.org/abs/2211.10438) outlier discussion",
    failure_mode="false_negative",
    mechanism="The scale should track the current batch's max magnitude. The buggy kernel "
               "instead uses a stale exponential-moving-average (EMA) scale. On mild Gaussian "
               "test data the EMA tracks the true max closely and the bug is invisible; a "
               "realistic outlier channel that appears only in the current batch is clipped "
               "hard by the stale EMA scale.",
    expected={
        "ground_truth": "kernel.py quantizes with a stale EMA scale instead of the current-batch max",
        "naive_allclose_verdict": "PASS on mild Gaussian data, FAIL once an outlier value appears",
        "correct_verdict": "BUGGY",
    },
    note="test.py quantizes/dequantizes mild N(0,1) data (passes) and then data containing one "
         "large outlier value the EMA has not caught up to (fails).",
    problem_txt='''"""FN5: INT8 quantization scale contract.

Spec: `quantize(x)` must compute scale = max(|x|) / 127 from the CURRENT
batch, quantize to int8, and dequantize as q * scale.

Reference implementation (ground truth):
''' + '''
import torch

def quant_dequant_reference(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().max() / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return q * scale
''',
    kernel_py='''import torch

# Stale scale from historical batches (simulating an EMA that has not
# adapted to the current batch's outlier).
_STALE_EMA_ABSMAX = 4.0


def quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """Candidate implementation. BUG: uses a stale EMA-derived scale instead
    of the current batch's max(|x|)."""
    scale = _STALE_EMA_ABSMAX / 127.0  # BUG: should be x.abs().max() / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return q * scale
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import quant_dequant  # candidate (buggy)


def quant_dequant_reference(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().max() / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return q * scale


def test_kernel():
    torch.manual_seed(0)

    # Mild case: current batch's max magnitude happens to match the stale EMA
    # exactly, so both scales coincide and the bug produces zero difference
    # -- exactly what ordinary, stable-distribution tests look like.
    x_mild = torch.randn(1024)
    x_mild = x_mild / x_mild.abs().max() * 4.0
    ref_mild = quant_dequant_reference(x_mild)
    cand_mild = quant_dequant(x_mild)
    mild_pass = torch.allclose(cand_mild, ref_mild, rtol=1e-2, atol=1e-2)

    # Adversarial case: current batch has one large outlier the EMA hasn't seen.
    x_outlier = torch.randn(1024) * 1.0
    x_outlier[0] = 50.0
    ref_outlier = quant_dequant_reference(x_outlier)
    cand_outlier = quant_dequant(x_outlier)
    outlier_pass = torch.allclose(cand_outlier, ref_outlier, rtol=1e-2, atol=1e-2)

    print(f"mild data allclose = {mild_pass} (bug hidden)")
    print(f"outlier data allclose = {outlier_pass}, reconstructed outlier = {cand_outlier[0].item():.2f} vs true {x_outlier[0].item():.2f}")
    print("FN DEMONSTRATED" if mild_pass and not outlier_pass else "tune constants")
    return mild_pass and not outlier_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FN6: low-bit recurrent state accumulation drift
# ---------------------------------------------------------------------------
add(
    name="fn6_lowbit_recurrence_drift",
    group="FN", seed_class="FN6",
    kernel_family="Low-bit KV-cache-style recurrent state compression",
    reference="TurboQuant-style extreme KV-cache compression (https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)",
    failure_mode="false_negative",
    mechanism="Each recurrence step quantizes the running state to a coarse grid, introducing "
               "a small per-step rounding error. A short rollout accumulates too little drift "
               "to matter; a long rollout compounds the same per-step error into a large "
               "deviation from the exact FP32 recurrence.",
    expected={
        "ground_truth": "kernel.py's per-step quantization introduces a real, compounding bias",
        "naive_allclose_verdict": "PASS after 5 steps, FAIL after 2000 steps",
        "correct_verdict": "BUGGY",
    },
    note="test.py runs the same recurrence for T=5 and T=2000 steps and compares final state "
         "against an exact FP32 recurrence.",
    problem_txt='''"""FN6: recurrent state update contract.

Spec: `step(state, x, decay)` must implement the exact recurrence
state' = decay * state + x in full precision.

Reference implementation (ground truth):
''' + '''
import torch

def rollout_reference(x_seq: torch.Tensor, decay: float) -> torch.Tensor:
    state = torch.zeros(())
    for x in x_seq:
        state = decay * state + x
    return state
''',
    kernel_py='''import torch

_QUANT_STEP = 5e-3  # coarse quantization grid for the compressed state


def _quantize_state(state: torch.Tensor) -> torch.Tensor:
    return torch.round(state / _QUANT_STEP) * _QUANT_STEP


def rollout(x_seq: torch.Tensor, decay: float) -> torch.Tensor:
    """Candidate implementation. BUG: rounds the state to a coarse grid every
    step, introducing a small error that compounds over long rollouts."""
    state = torch.zeros(())
    for x in x_seq:
        state = _quantize_state(decay * state + x)
    return state
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import rollout  # candidate (buggy)


def rollout_reference(x_seq: torch.Tensor, decay: float) -> torch.Tensor:
    state = torch.zeros(())
    for x in x_seq:
        state = decay * state + x
    return state


def test_kernel():
    torch.manual_seed(0)
    decay = 0.999

    x_short = torch.randn(5) * 0.01
    ref_short = rollout_reference(x_short, decay)
    cand_short = rollout(x_short, decay)
    short_pass = torch.allclose(cand_short, ref_short, rtol=1e-2, atol=1e-2)

    x_long = torch.randn(2000) * 0.01
    ref_long = rollout_reference(x_long, decay)
    cand_long = rollout(x_long, decay)
    long_pass = torch.allclose(cand_long, ref_long, rtol=1e-2, atol=1e-2)

    print(f"T=5: allclose={short_pass}, diff={(cand_short-ref_short).abs().item():.6f}")
    print(f"T=2000: allclose={long_pass}, diff={(cand_long-ref_long).abs().item():.6f}")
    print("FN DEMONSTRATED" if short_pass and not long_pass else "tune constants")
    return short_pass and not long_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FN7: RMSNorm eps placement
# ---------------------------------------------------------------------------
add(
    name="fn7_rmsnorm_eps_placement",
    group="FN", seed_class="FN7",
    kernel_family="RMSNorm",
    reference="https://arxiv.org/abs/1910.07467",
    failure_mode="false_negative",
    mechanism="The formula is x / sqrt(mean(x^2) + eps). The buggy kernel computes "
               "x / (sqrt(mean(x^2)) + eps) instead. At normal magnitudes both placements are "
               "numerically indistinguishable; only a near-zero row (e.g. padding) makes the "
               "two denominators differ by orders of magnitude.",
    expected={
        "ground_truth": "kernel.py adds eps outside the sqrt instead of inside",
        "naive_allclose_verdict": "PASS on normal rows, FAIL on a near-zero row",
        "correct_verdict": "BUGGY",
    },
    note="test.py compares normal random rows (passes) and a near-zero row (fails, output "
         "magnitude off by ~3 orders of magnitude).",
    problem_txt='''"""FN7: RMSNorm epsilon placement contract.

Spec: rmsnorm(x) = x / sqrt(mean(x**2) + eps).

Reference implementation (ground truth):
''' + '''
import torch

def rmsnorm_reference(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
''',
    kernel_py='''import torch


def rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Candidate implementation. BUG: adds eps outside the sqrt."""
    return x / (torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True)) + eps)
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import rmsnorm  # candidate (buggy)


def rmsnorm_reference(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def test_kernel():
    torch.manual_seed(0)

    x_normal = torch.randn(8, 16)
    ref_normal = rmsnorm_reference(x_normal)
    cand_normal = rmsnorm(x_normal)
    normal_pass = torch.allclose(cand_normal, ref_normal, rtol=1e-2, atol=1e-2)

    x_tiny = torch.full((1, 16), 2e-8)
    ref_tiny = rmsnorm_reference(x_tiny)
    cand_tiny = rmsnorm(x_tiny)
    tiny_pass = torch.allclose(cand_tiny, ref_tiny, rtol=1e-2, atol=1e-2)

    print(f"normal rows: allclose={normal_pass}")
    print(f"near-zero row: allclose={tiny_pass}, ref_scale={ref_tiny.abs().max().item():.3e}, cand_scale={cand_tiny.abs().max().item():.3e}")
    print("FN DEMONSTRATED" if normal_pass and not tiny_pass else "tune constants")
    return normal_pass and not tiny_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FP1a: NSA top-k unspecified tie-break -> valid alt rejected
# ---------------------------------------------------------------------------
add(
    name="fp1_topk_tie_unspecified",
    group="FP", seed_class="FP1",
    kernel_family="NSA-style top-k block selection (unspecified tie-break)",
    reference="https://arxiv.org/abs/2502.11089 (Native Sparse Attention, selection branch)",
    failure_mode="false_positive",
    mechanism="Unlike fn1_topk_tieblock_dilution, here the spec does NOT define a winner for "
               "exact ties -- any of the tied candidates is a valid top-k member. With k=1 and "
               "no averaging to dilute the difference, two equally valid implementations that "
               "break the tie differently produce very different downstream outputs, so naive "
               "allclose against one arbitrary reference wrongly reports a failure.",
    expected={
        "ground_truth": "kernel.py is an equally valid tie-break choice, not a bug",
        "naive_allclose_verdict": "FAIL (wrongly rejects a valid implementation)",
        "correct_verdict": "CORRECT (a permitted alternative)",
    },
    note="test.py ties the single top score between two candidates with very different value "
         "vectors and k=1, so the two valid choices disagree hugely on raw allclose.",
    problem_txt='''"""FP1: top-1 selection with UNSPECIFIED tie-breaking.

Spec: `select_top1(scores)` must return the index of the highest-scoring
candidate. The spec places NO requirement on which index wins an exact tie
-- any tied candidate is a valid answer.

Reference implementation A (one valid choice; lower index wins ties):
''' + '''
import torch

def select_top1_reference(scores: torch.Tensor) -> torch.Tensor:
    return torch.argsort(-scores, dim=-1, stable=True)[..., 0]
''',
    kernel_py='''import torch


def select_top1(scores: torch.Tensor) -> torch.Tensor:
    """Candidate implementation B: equally valid, breaks ties toward the
    HIGHER index instead. Not a bug -- the contract leaves this unspecified."""
    n = scores.shape[-1]
    reversed_order = torch.argsort(-scores.flip(-1), dim=-1, stable=True)[..., 0]
    return n - 1 - reversed_order
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import select_top1  # candidate (valid alternative)


def select_top1_reference(scores: torch.Tensor) -> torch.Tensor:
    return torch.argsort(-scores, dim=-1, stable=True)[..., 0]


def test_kernel():
    scores = torch.zeros(1, 8)
    scores[0, 2] = 5.0
    scores[0, 6] = 5.0  # exact tie for the top score, unspecified who wins

    ref_idx = select_top1_reference(scores)
    cand_idx = select_top1(scores)
    assert ref_idx.item() == 2
    assert cand_idx.item() == 6

    values = torch.zeros(8, 4)
    values[2] = 0.0
    values[6] = 10.0  # deliberately very different so raw allclose blows up

    ref_out = values[ref_idx]
    cand_out = values[cand_idx]

    naive_pass = torch.allclose(cand_out, ref_out, rtol=1e-2, atol=1e-2)
    both_valid = ref_idx.item() in (2, 6) and cand_idx.item() in (2, 6)
    print(f"ref picked {ref_idx.item()}, candidate picked {cand_idx.item()}; both are valid top-1 choices under the spec")
    print(f"naive allclose = {naive_pass} (expected False: valid impl wrongly rejected)")
    print("FP DEMONSTRATED" if (not naive_pass) and both_valid else "tune constants")
    return (not naive_pass) and both_valid


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FP1b: quantized MoE router tie at FP16 precision
# ---------------------------------------------------------------------------
add(
    name="fp1_moe_router_tie_fp16",
    group="FP", seed_class="FP1",
    kernel_family="Quantized MoE top-2 router",
    reference="vLLM-style MoE routing (https://github.com/vllm-project/vllm)",
    failure_mode="false_positive",
    mechanism="Two FP16 expert logits round to the exact same value, creating an unspecified "
               "tie for the second routed expert. Two equally valid tie-break rules pick "
               "different experts; because experts are independently parameterized, the final "
               "weighted output differs well beyond any reasonable tolerance even though "
               "neither implementation violates the (silent-on-ties) contract.",
    expected={
        "ground_truth": "kernel.py picks a different, equally valid second expert on a tie",
        "naive_allclose_verdict": "FAIL (wrongly rejects a valid implementation)",
        "correct_verdict": "CORRECT (a permitted alternative)",
    },
    note="test.py rounds two expert logits to the same FP16 value, computes the top-2 weighted "
         "output both ways, and shows the raw outputs diverge substantially.",
    problem_txt='''"""FP1 (extension): quantized MoE top-2 routing with an FP16 logit tie.

Spec: `route_top2(logits)` returns the indices of the two highest-scoring
experts. The contract is silent on which expert wins when two FP16-rounded
logits are exactly equal.

Reference implementation A (lower index wins the tie):
''' + '''
import torch

def route_top2_reference(logits_fp16: torch.Tensor) -> torch.Tensor:
    return torch.argsort(-logits_fp16.float(), stable=True)[:2]
''',
    kernel_py='''import torch


def route_top2(logits_fp16: torch.Tensor) -> torch.Tensor:
    """Candidate implementation B: equally valid, higher index wins the tie."""
    n = logits_fp16.shape[-1]
    reversed_order = torch.argsort(-logits_fp16.float().flip(-1), stable=True)
    flipped_back = n - 1 - reversed_order
    return flipped_back[:2]
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import route_top2  # candidate (valid alternative)


def route_top2_reference(logits_fp16: torch.Tensor) -> torch.Tensor:
    return torch.argsort(-logits_fp16.float(), stable=True)[:2]


def test_kernel():
    torch.manual_seed(0)
    n_experts, d = 6, 8
    logits = torch.tensor([9.0, 3.0001, 3.0, 1.0, 0.5, 0.1], dtype=torch.float32)
    logits_fp16 = logits.half()  # 3.0001 and 3.0 round to the same fp16 value -> exact tie
    assert logits_fp16[1] == logits_fp16[2]

    expert_out = torch.randn(n_experts, d)

    ref_idx = route_top2_reference(logits_fp16)
    cand_idx = route_top2(logits_fp16)
    assert set(ref_idx.tolist()) == {0, 1} or set(ref_idx.tolist()) == {0, 2}
    assert ref_idx.tolist() != cand_idx.tolist(), "the two valid tie-breaks must actually differ"

    ref_out = expert_out[ref_idx].mean(dim=0)
    cand_out = expert_out[cand_idx].mean(dim=0)

    naive_pass = torch.allclose(cand_out, ref_out, rtol=1e-2, atol=1e-2)
    print(f"ref experts={ref_idx.tolist()}, candidate experts={cand_idx.tolist()} (both valid under an unspecified tie)")
    print(f"naive allclose = {naive_pass} (expected False)")
    print("FP DEMONSTRATED" if not naive_pass else "tune constants")
    return not naive_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FP2a: RoPE GPT-NeoX vs GPT-J layout convention
# ---------------------------------------------------------------------------
add(
    name="fp2_rope_neox_vs_gptj",
    group="FP", seed_class="FP2",
    kernel_family="Rotary Position Embeddings (RoPE)",
    reference="https://arxiv.org/abs/2104.09864",
    failure_mode="false_positive",
    mechanism="GPT-NeoX-style RoPE (rotate_half on two contiguous halves) and GPT-J-style RoPE "
               "(rotate interleaved adjacent pairs) are equivalent up to a fixed permutation of "
               "the feature dimension. Comparing their raw rotated tensors element-by-element "
               "fails almost everywhere, even though the resulting attention scores -- the "
               "quantity that actually matters -- agree once each convention is applied "
               "consistently to both query and key.",
    expected={
        "ground_truth": "kernel.py is a mathematically equivalent RoPE layout, not a bug",
        "naive_allclose_verdict": "FAIL on raw rotated tensors",
        "correct_verdict": "CORRECT (downstream attention scores match within tolerance)",
    },
    note="test.py shows raw-tensor allclose fails, but the attention score q.k computed "
         "consistently within each convention matches closely.",
    problem_txt='''"""FP2: RoPE layout convention equivalence.

Spec: apply rotary position embeddings to q/k so that the dot product of a
rotated query and key encodes their relative position. The GPT-NeoX layout
(split into two halves) and the GPT-J layout (interleaved pairs) are both
valid as long as each is applied consistently to both q and k.

Reference implementation (GPT-NeoX / half-split layout):
''' + '''
import torch

def rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin
''',
    kernel_py='''import torch


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Candidate implementation: GPT-J / interleaved-pair layout. A valid,
    different RoPE convention -- not a bug."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + rotated * sin
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import rope  # candidate (different valid layout)


def rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def make_cos_sin(seq_len, d, base=10000.0, interleaved=False):
    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)  # [seq_len, half]
    if interleaved:
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
    else:
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def to_interleaved(x: torch.Tensor, half: int) -> torch.Tensor:
    """Re-lay-out a half-split vector [x1(0..half-1), x2(0..half-1)] as the
    interleaved pairing [x1[0], x2[0], x1[1], x2[1], ...] used by GPT-J RoPE.
    This is the explicit permutation connecting the two conventions."""
    out = torch.empty_like(x)
    out[..., 0::2] = x[..., :half]
    out[..., 1::2] = x[..., half:]
    return out


def test_kernel():
    torch.manual_seed(0)
    seq_len, d = 4, 8
    half = d // 2
    q = torch.randn(seq_len, d)
    k = torch.randn(seq_len, d)

    cos_half, sin_half = make_cos_sin(seq_len, d, interleaved=False)
    cos_int, sin_int = make_cos_sin(seq_len, d, interleaved=True)

    q_ref = rope_reference(q, cos_half, sin_half)
    k_ref = rope_reference(k, cos_half, sin_half)

    # Feed the SAME underlying q/k, re-laid-out into the interleaved
    # convention via the explicit permutation, so both kernels rotate
    # mathematically corresponding pairs.
    q_cand = rope(to_interleaved(q, half), cos_int, sin_int)
    k_cand = rope(to_interleaved(k, half), cos_int, sin_int)

    raw_pass = torch.allclose(q_cand, q_ref, rtol=1e-2, atol=1e-2)

    scores_ref = q_ref @ k_ref.T
    scores_cand = q_cand @ k_cand.T
    downstream_pass = torch.allclose(scores_cand, scores_ref, rtol=1e-2, atol=1e-2)

    print(f"raw rotated-tensor allclose = {raw_pass} (expected False)")
    print(f"downstream attention-score allclose = {downstream_pass} (expected True: both conventions are valid)")
    print("FP DEMONSTRATED" if (not raw_pass) and downstream_pass else "tune constants")
    return (not raw_pass) and downstream_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FP3a: online vs two-pass cross-entropy reduction order
# ---------------------------------------------------------------------------
add(
    name="fp3_ce_reduction_order",
    group="FP", seed_class="FP3",
    kernel_family="Fused cross-entropy over a large vocabulary",
    reference="https://github.com/linkedin/Liger-Kernel",
    failure_mode="false_positive",
    mechanism="A one-pass online-softmax cross-entropy and a conventional two-pass "
               "implementation are mathematically identical algorithms that only differ in "
               "floating-point summation order. With a large FP16 vocabulary, a handful of "
               "per-token losses can differ enough to fail a strict per-element tolerance even "
               "though the batch-mean loss (what actually matters) matches closely.",
    expected={
        "ground_truth": "kernel.py is an equivalent one-pass algorithm, not a bug",
        "naive_allclose_verdict": "FAIL on some per-token elements under a strict per-element check",
        "correct_verdict": "CORRECT (batch-mean loss matches within tolerance)",
    },
    note="test.py compares per-token FP16 cross-entropy losses element-wise (can fail on a "
         "few tokens) versus the batch mean (matches).",
    problem_txt='''"""FP3: cross-entropy reduction-order equivalence.

Spec: `cross_entropy(logits, target)` returns per-token negative log-likelihood.
A two-pass (max, then sum-exp) and a one-pass online-softmax algorithm are both
valid, exact (non-approximating) implementations that only differ in
floating-point summation order.

Reference implementation (two-pass):
''' + '''
import torch

def cross_entropy_reference(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    m = logits.max(dim=-1, keepdim=True).values
    lse = (logits - m).exp().sum(dim=-1).log() + m.squeeze(-1)
    return lse - logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
''',
    kernel_py='''import torch


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Candidate implementation: one-pass online-softmax cross-entropy in
    fp16 intermediate precision. Mathematically equivalent to the two-pass
    reference; differs only in floating-point summation order and rounding."""
    n_tokens, vocab = logits.shape
    losses = torch.empty(n_tokens)
    for t in range(n_tokens):
        row = logits[t].half()  # simulate a lower-precision fused kernel
        running_max = torch.tensor(float("-inf"), dtype=torch.float16)
        running_sum = torch.tensor(0.0, dtype=torch.float16)
        for v in range(vocab):
            x = row[v]
            new_max = torch.maximum(running_max, x)
            # Accumulate in fp16 at every step, like a fused low-precision
            # kernel would -- this is what actually differs from the
            # reference's fp32 two-pass reduction order.
            running_sum = (running_sum * torch.exp(running_max - new_max) + torch.exp(x - new_max)).half()
            running_max = new_max
        lse = (running_max + torch.log(running_sum)).float()
        losses[t] = lse - row[target[t]].float()
    return losses
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import cross_entropy  # candidate (equivalent, different reduction order)


def cross_entropy_reference(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    m = logits.max(dim=-1, keepdim=True).values
    lse = (logits - m).exp().sum(dim=-1).log() + m.squeeze(-1)
    return lse - logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)


def test_kernel():
    torch.manual_seed(0)
    n_tokens, vocab = 2, 20000
    logits = torch.randn(n_tokens, vocab) * 3.0
    target = torch.randint(0, vocab, (n_tokens,))

    ref = cross_entropy_reference(logits, target)
    cand = cross_entropy(logits, target)

    per_token_pass = torch.allclose(cand, ref, rtol=1e-3, atol=1e-3)
    mean_pass = torch.allclose(cand.mean(), ref.mean(), rtol=1e-2, atol=1e-2)

    print(f"strict per-token allclose (rtol=1e-3) = {per_token_pass} (may fail due to fp16 rounding)")
    print(f"batch-mean allclose = {mean_pass} (expected True: both are valid, equivalent algorithms)")
    print("FP DEMONSTRATED" if (not per_token_pass) and mean_pass else "informational: reduction-order drift may be smaller than expected on this seed")
    return mean_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FP3b: parallel tree-sum vs serial sum accumulation drift
# ---------------------------------------------------------------------------
add(
    name="fp3_parallel_vs_serial_scan",
    group="FP", seed_class="FP3",
    kernel_family="Mamba2/SSD-style chunked parallel scan vs serial recurrence",
    reference="https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py",
    failure_mode="false_positive",
    mechanism="A tree-shaped parallel reduction and a left-to-right serial sum are both exact "
               "algorithms for the same total, differing only in floating-point addition "
               "order. As the sequence length grows, the accumulated rounding drift between "
               "the two orders grows large enough to fail a tolerance calibrated on short "
               "sequences, even though both are correct.",
    expected={
        "ground_truth": "kernel.py is an exact, equivalent reduction order, not a bug",
        "naive_allclose_verdict": "FAIL on long sequences under a tight, short-sequence-calibrated tolerance",
        "correct_verdict": "CORRECT (drift matches the expected floating-point accumulation signature)",
    },
    note="test.py sums the same FP16 sequence via serial accumulation and via a tree reduction, "
         "at short and long lengths, showing drift grows with length.",
    problem_txt='''"""FP3 (extension): reduction-order equivalence for a running-sum scan.

Spec: `total_sum(x)` returns the exact sum of x. A serial left-to-right
accumulation and a tree-shaped parallel reduction are both exact algorithms
that only differ in floating-point addition order.

Reference implementation (serial accumulation):
''' + '''
import torch

def total_sum_reference(x: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros((), dtype=x.dtype)
    for v in x:
        acc = acc + v
    return acc
''',
    kernel_py='''import torch


def total_sum(x: torch.Tensor) -> torch.Tensor:
    """Candidate implementation: tree-shaped parallel reduction. Exact and
    equivalent to the serial reference; differs only in addition order."""
    vals = list(x.unbind(0))
    while len(vals) > 1:
        nxt = []
        for i in range(0, len(vals) - 1, 2):
            nxt.append(vals[i] + vals[i + 1])
        if len(vals) % 2 == 1:
            nxt.append(vals[-1])
        vals = nxt
    return vals[0]
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import total_sum  # candidate (equivalent, different reduction order)


def total_sum_reference(x: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros((), dtype=x.dtype)
    for v in x:
        acc = acc + v
    return acc


def test_kernel():
    torch.manual_seed(0)

    def make_seq(n_small):
        # Classic magnitude-disparity pattern: many small terms bracketed by
        # one large term. A serial left-to-right sum loses precision on the
        # small terms once the running total is dominated by the large one;
        # a tree reduction sums the small terms together first, preserving
        # more of their precision. Both are exact, valid summation orders.
        small = torch.ones(n_small)
        return torch.cat([small[: n_small // 2], torch.tensor([2000.0]), small[n_small // 2 :]]).half()

    x_short = make_seq(8)
    ref_short = total_sum_reference(x_short)
    cand_short = total_sum(x_short)
    short_diff = (cand_short.float() - ref_short.float()).abs().item()

    x_long = make_seq(20000)
    ref_long = total_sum_reference(x_long)
    cand_long = total_sum(x_long)
    long_diff = (cand_long.float() - ref_long.float()).abs().item()
    exact_long = 2000.0 + 20000.0

    print(f"short sequence (n=8+1): serial={ref_short.item():.2f}, tree={cand_short.item():.2f}, diff={short_diff:.4f}")
    print(f"long sequence (n=20000+1): exact={exact_long:.1f}, serial={ref_long.item():.2f}, tree={cand_long.item():.2f}, diff={long_diff:.4f}")
    print("FP DEMONSTRATED" if long_diff > short_diff and long_diff > 1.0 else "informational: fp16 drift magnitude depends on backend rounding")
    return long_diff > short_diff


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FP4: FP8-style quantization noise vs an FP32-calibrated tolerance
# ---------------------------------------------------------------------------
add(
    name="fp4_fp8_quant_tolerance",
    group="FP", seed_class="FP4",
    kernel_family="Low-bit (FP8-style) attention/GEMM quantization noise",
    reference="Attn-QAT-style 4-bit attention (illustrative low-bit precision model)",
    failure_mode="false_positive",
    mechanism="A correct low-bit kernel intentionally rounds values to a coarse grid, "
               "introducing quantization noise that is part of its design, not a defect. "
               "Checking it with a tolerance calibrated for FP32/FP16 kernels causes a "
               "correct low-bit kernel to fail, even though its error is exactly the expected "
               "magnitude for its declared precision.",
    expected={
        "ground_truth": "kernel.py is a correct low-bit kernel; its error matches the declared quantization step",
        "naive_allclose_verdict": "FAIL under an FP32-calibrated tolerance (rtol=1e-2, atol=1e-2)",
        "correct_verdict": "CORRECT (passes under a tolerance derived from the quantization step)",
    },
    note="test.py compares an FP32 reference matmul against a simulated FP8-precision matmul "
         "(values rounded to a coarse grid before multiplying), showing the expected error "
         "exceeds the FP32 tolerance but fits a format-aware tolerance.",
    problem_txt='''"""FP4: precision-aware tolerance contract.

Spec: `quantized_matmul(a, b, bits)` computes a @ b after rounding a and b to
a `bits`-bit signed grid scaled by their max magnitude. The declared
precision determines how much quantization error is expected and correct.

Reference implementation (full precision):
''' + '''
import torch

def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b
''',
    kernel_py='''import torch


def _quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    levels = 2 ** (bits - 1) - 1
    scale = x.abs().max() / levels
    q = torch.clamp(torch.round(x / scale), -levels, levels)
    return q * scale


def quantized_matmul(a: torch.Tensor, b: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Candidate implementation: correct FP8-style (bits=8) quantized matmul.
    Its numerical error is the intended cost of low-bit precision, not a bug."""
    a_q = _quantize(a, bits)
    b_q = _quantize(b, bits)
    return a_q @ b_q
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import quantized_matmul, _quantize  # candidate (correct, low-bit)


def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def test_kernel():
    torch.manual_seed(0)
    a = torch.randn(32, 32)
    b = torch.randn(32, 32)

    ref = matmul_reference(a, b)
    cand = quantized_matmul(a, b, bits=8)

    fp32_tolerance_pass = torch.allclose(cand, ref, rtol=1e-2, atol=1e-2)

    # format-aware tolerance derived from the actual quantization step size
    a_step = a.abs().max() / (2 ** 7 - 1)
    b_step = b.abs().max() / (2 ** 7 - 1)
    expected_abs_err = 32 * (a_step * b.abs().mean() + b_step * a.abs().mean())  # rough error bound for a 32-dim dot product
    format_aware_pass = (cand - ref).abs().max().item() <= 3 * expected_abs_err.item()

    print(f"FP32-calibrated allclose(rtol=1e-2, atol=1e-2) = {fp32_tolerance_pass} (expected False)")
    print(f"format-aware bound check (~3x expected quantization error) = {format_aware_pass} (expected True)")
    print("FP DEMONSTRATED" if (not fp32_tolerance_pass) and format_aware_pass else "tune constants")
    return (not fp32_tolerance_pass) and format_aware_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FP5: stochastic rounding
# ---------------------------------------------------------------------------
add(
    name="fp5_stochastic_rounding_bf16",
    group="FP", seed_class="FP5",
    kernel_family="Stochastic-rounding FP32->BF16 cast for training weight updates",
    reference="Stochastic rounding in low-precision training (general technique, e.g. used in BF16/FP8 optimizer state updates)",
    failure_mode="false_positive",
    mechanism="The kernel intentionally rounds up or down randomly, weighted by proximity to "
               "each representable low-precision value, so two correct runs of the exact same "
               "code produce different elementwise outputs on the same input. Treating one "
               "run's output as the only valid reference for another run is a false positive "
               "on intended nondeterminism.",
    expected={
        "ground_truth": "kernel.py is a correct stochastic-rounding cast; run-to-run variation is intended",
        "naive_allclose_verdict": "FAIL when comparing two independent runs elementwise",
        "correct_verdict": "CORRECT (the mean rounding bias across many runs matches the exact value)",
    },
    note="test.py runs the same stochastic-rounding cast twice with different seeds (elementwise "
         "mismatch), then checks the mean over many runs converges to the exact FP32 value "
         "(unbiased).",
    problem_txt='''"""FP5: stochastic rounding contract.

Spec: `stochastic_round_to_bf16(x)` casts x to a value representable at
bf16-like granularity (step size `STEP`), rounding UP with probability equal
to how close x is to the upper grid point, so that E[output] == x. Two runs
on the same input need not match elementwise; the contract is on the
distribution, not on any single realization.

Reference (illustrative "typical" run, used only for the single-run check):
''' + '''
import torch

STEP = 0.05

def deterministic_round_reference(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest -- a DIFFERENT, deterministic policy, shown only to
    contrast with the stochastic contract; not the ground truth being tested."""
    return torch.round(x / STEP) * STEP
''',
    kernel_py='''import torch

STEP = 0.05


def stochastic_round_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """Candidate implementation: correct stochastic rounding. E[output] == x,
    but any single realization is intentionally random."""
    lower = torch.floor(x / STEP) * STEP
    upper = lower + STEP
    p_up = (x - lower) / STEP
    up_mask = torch.rand_like(x) < p_up
    return torch.where(up_mask, upper, lower)
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import stochastic_round_to_bf16, STEP  # candidate (correct, intentionally random)


def test_kernel():
    torch.manual_seed(0)
    x = torch.full((2048,), 0.137)  # exact value we expect to be preserved in expectation

    torch.manual_seed(1)
    run_a = stochastic_round_to_bf16(x)
    torch.manual_seed(2)
    run_b = stochastic_round_to_bf16(x)

    naive_pass = torch.allclose(run_a, run_b, rtol=1e-2, atol=1e-2)

    mean_estimate = run_a.mean()
    unbiased_pass = torch.allclose(mean_estimate, x[0], rtol=0.0, atol=2 * STEP / (2048 ** 0.5) * 4)

    print(f"two independent runs, elementwise allclose = {naive_pass} (expected False)")
    print(f"mean over {x.numel()} draws = {mean_estimate.item():.5f} vs true value {x[0].item():.5f}; unbiased check = {unbiased_pass}")
    print("FP DEMONSTRATED" if (not naive_pass) and unbiased_pass else "tune constants")
    return (not naive_pass) and unbiased_pass


if __name__ == "__main__":
    test_kernel()
''',
)

# ---------------------------------------------------------------------------
# FP6: relative error unstable near zero
# ---------------------------------------------------------------------------
add(
    name="fp6_relative_error_near_zero",
    group="FP", seed_class="FP6",
    kernel_family="Any continuous kernel whose output can approach zero (e.g. masked attention weight)",
    reference="General instability of relative error near zero; motivates torch.allclose's atol+rtol design",
    failure_mode="false_positive",
    mechanism="Both implementations are correct; the true value is extremely small (near the "
               "floor of normal floating-point rounding). A negligible absolute difference "
               "becomes a huge relative error once divided by the tiny reference magnitude, "
               "so a relative-error-only check flags a nonexistent problem.",
    expected={
        "ground_truth": "kernel.py is numerically correct; the discrepancy is ordinary rounding noise",
        "naive_allclose_verdict": "FAIL under a relative-error-only metric",
        "correct_verdict": "CORRECT (passes under torch.allclose's combined atol+rtol, which is designed for exactly this)",
    },
    note="test.py compares two correct computations of a value near 1e-8 using a relative-only "
         "metric (fails) and torch.allclose with both atol and rtol (passes).",
    problem_txt='''"""FP6: comparison-metric stability near zero.

Spec: `masked_softmax_weight(logits, mask)` returns the softmax weight of a
heavily masked (near-zero) position. Both implementations below compute it
correctly; they differ only by ordinary floating-point rounding of order
1e-9, which is irrelevant at this magnitude.

Reference implementation:
''' + '''
import torch

def masked_softmax_weight_reference(logits: torch.Tensor, idx: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return probs[idx]
''',
    kernel_py='''import torch


def masked_softmax_weight(logits: torch.Tensor, idx: int) -> torch.Tensor:
    """Candidate implementation: mathematically identical computation via the
    log-sum-exp identity, correct up to ordinary floating-point rounding."""
    m = logits.max()
    lse = (logits - m).exp().sum().log() + m
    return torch.exp(logits[idx] - lse)
''',
    test_py='''import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import masked_softmax_weight  # candidate (correct, tiny rounding diff)


def masked_softmax_weight_reference(logits: torch.Tensor, idx: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return probs[idx]


def test_kernel():
    torch.manual_seed(0)
    logits = torch.tensor([20.0, 20.0, 20.0, -0.5])  # index 3 gets a tiny softmax weight
    idx = 3

    ref = masked_softmax_weight_reference(logits, idx)
    cand = masked_softmax_weight(logits, idx)

    abs_diff = (cand - ref).abs().item()
    relative_only_pass = abs_diff <= 1e-2 * ref.abs().item()  # naive relative-only metric
    combined_pass = torch.allclose(cand, ref, rtol=1e-2, atol=1e-6)  # torch.allclose's real formula

    print(f"true value ~ {ref.item():.3e}, abs diff = {abs_diff:.3e}")
    print(f"relative-error-only check = {relative_only_pass} (can fail even though both are correct)")
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
        "benchmark_version": "fn_fp_v1",
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
        "source": "handwritten_synthetic_fn_fp",
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
