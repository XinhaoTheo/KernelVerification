import torch, json
from kernel import sample_recovered_tokens

torch.manual_seed(0)
dev = 'cuda'
B, V = 1, 1024
target = torch.full((B, V), 0.05, device=dev)
draft = target.clone()
inv_q = torch.full((B, V), 1.0, device=dev)
# token 10: small positive residual with modest inv_q
draft[0, 10] = 0.04
inv_q[0, 10] = 2.0  # score = 0.01*2 = 0.02
# token 500: strong negative residual with huge inv_q -> unclamped score 0.4 beats 0.02
draft[0, 500] = 0.45
inv_q[0, 500] = 1e9  # score = (0.05-0.45)*1e9 = -4e8? wait, negative*positive is negative
# fix: to make negative residual win over positive score, need negative residual * large inv_q POSITIVE is impossible; the risk is that a large positive-residual? Re-check: unclamped score of over-proposed token is negative, can't beat positive. But claim states negative * large inv_q could exceed small positive score only if inv_q were negative or residual sign issue. Let's also test: positive residual tiny * inv_q negative? inv_q is 1/q > 0. So construct case: positive residual tiny (score ~0), over-proposed token residual negative tiny but inv_q huge -> score = -tiny*huge = very negative, still negative. Cannot beat 0. Instead test the realistic failure: positive residual scores are all 0? covered by c2. Try: token A residual +1e-9 (score ~1e-9), token B residual -1e-6 with inv_q huge -> score negative. No win. So probe whether kernel output ever equals an over-proposed token when positive-residual tokens exist with positive scores.
resid = (target - draft).clamp(min=0)
ref = (resid * inv_q).argmax(dim=1)
out = sample_recovered_tokens(target, draft, inv_q)
# extra targeted case: make the only positive residual have inv_q very small (score ~0) and an over-proposed token have tiny negative residual
target2 = torch.full((B, V), 0.0, device=dev); draft2 = target2.clone(); inv_q2 = torch.rand(B, V, device=dev) * 10 + 1.0
target2[0, 3] = 1.0; inv_q2[0, 3] = 1e-9  # score 1e-9
draft2[0, 700] = 0.5  # residual -0.5, inv_q ~5 -> score -2.5
ref2 = ((target2 - draft2).clamp(min=0) * inv_q2).argmax(dim=1)
out2 = sample_recovered_tokens(target2, draft2, inv_q2)
res = dict(
    case1_kernel=int(out[0].item()), case1_ref=int(ref[0].item()),
    case2_kernel=int(out2[0].item()), case2_ref=int(ref2[0].item()),
    case2_kernel_is_overproposed=bool(draft2[0, out2[0].item()] > target2[0, out2[0].item()]),
)
print(json.dumps(res))