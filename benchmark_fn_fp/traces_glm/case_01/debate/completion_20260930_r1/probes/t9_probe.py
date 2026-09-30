import torch, json
from kernel import sample_recovered_tokens

torch.manual_seed(0)
dev = 'cuda'
B, V = 2, 512
# row 0: every target <= draft (all residuals <= 0); some strictly negative
target = torch.rand(B, V, device=dev) * 0.4
draft = target + 0.5  # all residuals negative
inv_q = torch.rand(B, V, device=dev) * 5 + 0.5
# row 1: benign positive-residual control
target[1] = torch.rand(V, device=dev)
draft[1] = torch.rand(V, device=dev) * 0.5
ref = ((target - draft).clamp(min=0) * inv_q).argmax(dim=1)
out = sample_recovered_tokens(target, draft, inv_q)
resid = target - draft
res = dict(
    kernel_row0=int(out[0].item()), ref_row0=int(ref[0].item()),
    kernel_row0_overproposed=bool(draft[0, out[0].item()].item() > target[0, out[0].item()].item()),
    ref_row0_overproposed=bool(draft[0, ref[0].item()].item() > target[0, ref[0].item()].item()),
    kernel_row1=int(out[1].item()), ref_row1=int(ref[1].item()),
    row1_match=bool(out[1].item() == ref[1].item()),
    row0_match=bool(out[0].item() == ref[0].item()),
)
print(json.dumps(res))