import torch, json, sys
sys.path.insert(0, "/root/cases/case_01")
from kernel import sample_recovered_tokens

torch.manual_seed(0)
dev = "cuda"
B, V = 8, 1024
fails = []
for trial in range(50):
    # make draft sometimes over-propose: draft = softmax(randn), target = softmax(randn)
    draft = torch.softmax(torch.randn(B, V, device=dev), -1)
    target = torch.softmax(torch.randn(B, V, device=dev), -1)
    q = torch.distributions.Exponential(torch.ones(B, V, device=dev)).sample()
    inv_q = 1.0 / q
    out = sample_recovered_tokens(target, draft, inv_q)
    ref = torch.argmax(torch.clamp(target - draft, min=0) * inv_q, dim=-1)
    if not torch.equal(out.long(), ref.long()):
        fails.append((trial, out.tolist(), ref.tolist()))

# heavy over-proposal case: target tiny where draft large, inv_q huge on a negative-residual token
draft = torch.softmax(torch.randn(B, V, device=dev), -1)
target = torch.softmax(torch.randn(B, V, device=dev), -1)
target[:, :100] = 1e-9
q = torch.distributions.Exponential(torch.ones(B, V, device=dev)).sample()
inv_q = 1.0 / q
inv_q[:, :100] = 1e6  # huge weight on over-proposed tokens
out = sample_recovered_tokens(target, draft, inv_q)
ref = torch.argmax(torch.clamp(target - draft, min=0) * inv_q, dim=-1)
adversarial_ok = torch.equal(out.long(), ref.long())
neg_tokens = (target - draft)[:, :100] < 0
print(json.dumps({
    "metric": "exact argmax match vs clamp(target-draft,0)*inv_q",
    "random_trials_failed": len(fails),
    "fail_examples": fails[:3],
    "adversarial_overproposed_case_match": bool(adversarial_ok),
    "adversarial_negative_residual_tokens": int(neg_tokens.sum()),
    "out_adversarial": out.tolist(),
    "ref_adversarial": ref.tolist(),
}))