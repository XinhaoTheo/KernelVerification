"""Triton kernel under test: fn2_vllm_rejection_sampler_missing_clip."""
import torch
import triton
import triton.language as tl


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    inv_q_ptr,
    vocab_size,
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
        prob = target_prob - draft_prob
        inv_q = tl.load(inv_q_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_mask, other=0.0)
        score = prob * inv_q
        score = tl.where(vocab_mask, score, float("-inf"))
        local_max, local_id = tl.max(score, axis=0, return_indices=True)
        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id
    recovered_id = tl.minimum(recovered_id, vocab_size - 1)
    tl.store(output_token_ids_ptr + req_idx, recovered_id)


def sample_recovered_tokens(target_probs, draft_probs, inv_q):
    batch_size, vocab_size = target_probs.shape
    out = torch.empty(batch_size, dtype=torch.int64, device=target_probs.device)
    BLOCK_SIZE = triton.next_power_of_2(vocab_size)
    sample_recovered_tokens_kernel[(batch_size,)](out, draft_probs, target_probs, inv_q, vocab_size, BLOCK_SIZE=BLOCK_SIZE)
    return out
