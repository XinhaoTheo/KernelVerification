import torch, json, sys
sys.path.insert(0, "/root/cases/case_145")
import kernel

neg0 = torch.tensor([0.0], dtype=torch.float16).to(torch.float64)
neg0 = neg0.to(torch.float16) * -1  # ensures sign set below
# Build rows with earlier -0 and later +0
def mk(*vals):
    return torch.tensor(vals, dtype=torch.float16)

neg_zero = torch.zeros(1, dtype=torch.float16); neg_zero[0] = -0.0
pos_zero = torch.zeros(1, dtype=torch.float16)
assert torch.copysign(torch.ones(1), neg_zero).item() < 0

x = torch.stack([torch.cat([neg_zero, mk(-1.0, -2.0), pos_zero]),
                 torch.cat([neg_zero, pos_zero, pos_zero]),
                 torch.cat([mk(-5.0), neg_zero, pos_zero])]).to("cuda").contiguous()
out = kernel.run(x)
ref = kernel.reference(x.cpu())
result = {"kernel": out.cpu().tolist(), "ref": ref.tolist(),
          "signs": torch.copysign(torch.ones(x.shape, dtype=torch.float16), x.cpu()).view(x.shape[0],-1).tolist()}
print(json.dumps(result))