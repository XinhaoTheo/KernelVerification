import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import sample_recovered_tokens as sample_buggy

import triton
import triton.language as tl


@triton.jit
def _sample_recovered_tokens_kernel_ref(
    output_token_ids_ptr, draft_probs_ptr, target_probs_ptr, inv_q_ptr, vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    max_val = tl.full((), float("-inf"), tl.float32)
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size
        draft_prob = tl.load(draft_probs_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_mask, other=0.0)
        target_prob = tl.load(target_probs_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_mask, other=0.0)
        prob = tl.maximum(target_prob - draft_prob, 0.0)  # real kernel: clamp negatives
        inv_q = tl.load(inv_q_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_mask, other=0.0)
        score = prob * inv_q
        score = tl.where(vocab_mask, score, float("-inf"))
        local_max, local_id = tl.max(score, axis=0, return_indices=True)
        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id
    recovered_id = tl.minimum(recovered_id, vocab_size - 1)
    tl.store(output_token_ids_ptr + req_idx, recovered_id)


def sample_recovered_tokens_reference(target_probs, draft_probs, inv_q):
    batch_size, vocab_size = target_probs.shape
    out = torch.empty(batch_size, dtype=torch.int64, device=target_probs.device)
    BLOCK_SIZE = triton.next_power_of_2(vocab_size)
    _sample_recovered_tokens_kernel_ref[(batch_size,)](out, draft_probs, target_probs, inv_q, vocab_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    vocab = 32
    # Use a deterministic, equal Gumbel draw (inv_q=1 everywhere) for every
    # vocab slot: a legitimate input to the real formula (q=1 for all slots
    # is just one particular Exponential(1) realization), which lets us
    # construct an exact, reproducible boundary case instead of relying on
    # randomness to occasionally trigger the bug.
    inv_q = torch.ones(1, vocab, device=device)

    # Mild: one token has a clear positive residual (target > draft); clamp
    # never engages because the positive-diff token dominates the race
    # regardless of what happens to any negative-diff token.
    target_mild = torch.zeros(1, vocab, device=device)
    target_mild[0, 0], target_mild[0, 1] = 0.9, 0.05
    draft_mild = torch.zeros(1, vocab, device=device)
    draft_mild[0, 0], draft_mild[0, 1] = 0.85, 0.10  # token 1: draft over-proposes, but token 0 still wins
    ref_mild = sample_recovered_tokens_reference(target_mild, draft_mild, inv_q)
    cand_mild = sample_buggy(target_mild, draft_mild, inv_q)
    mild_agree = torch.equal(ref_mild, cand_mild)

    # Adversarial: draft over-proposes EVERY token (target <= draft
    # everywhere), so the correct (clamped) race is an exact all-zero tie
    # that resolves to index 0 (the real kernel's `local_max > max_val` never
    # fires past the first tile on an all-zero score, so it keeps index 0).
    # The buggy (unclamped) race instead picks whichever token draft
    # over-proposed LEAST -- placed here at index 5, not index 0.
    target_adv = torch.zeros(1, vocab, device=device)
    draft_adv = torch.zeros(1, vocab, device=device)
    for i in range(vocab):
        draft_adv[0, i] = 0.05
        target_adv[0, i] = 0.05 - 0.02  # every token over-proposed by 0.02 ...
    target_adv[0, 5] = 0.05 - 0.0001  # ... except token 5, over-proposed by only 0.0001
    ref_adv = sample_recovered_tokens_reference(target_adv, draft_adv, inv_q)
    cand_adv = sample_buggy(target_adv, draft_adv, inv_q)
    adv_agree = torch.equal(ref_adv, cand_adv)

    print(f"mild (a real positive-residual token exists): reference token={ref_mild.item()}, candidate token={cand_mild.item()}, agree={mild_agree}")
    print(f"adversarial (draft over-proposes every token): reference token={ref_adv.item()} (arbitrary tie-break), candidate token={cand_adv.item()} (picks least-over-proposed), agree={adv_agree}")
    print("FN DEMONSTRATED" if mild_agree and not adv_agree else "tune constants")
    # What a conventional CI test would conclude: 常规路径单次运行，拒绝很少发生
    print(f"NAIVE_ALLCLOSE_VERDICT: {mild_agree}")
    return mild_agree and not adv_agree


if __name__ == "__main__":
    test_kernel()
