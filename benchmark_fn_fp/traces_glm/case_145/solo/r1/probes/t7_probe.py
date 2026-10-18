import torch, json, sys
sys.path.insert(0, "/root/cases/case_145")
import kernel

def mk(*vals):
    return torch.tensor(vals, dtype=torch.float16)

neg_zero = torch.tensor([-0.0], dtype=torch.float16)
pos_zero = torch.tensor([0.0], dtype=torch.float16)
assert torch.copysign(torch.ones(1), neg_zero).item() < 0

# rows of length 4: row0 earlier -0 later +0 with negatives; row1 -0,+0,+0; row2 -5,-0,+0
x = torch.stack([
    torch.cat([neg_zero, mk(-1.0, -2.0), pos_zero]),
    torch.cat([neg_zero, pos_zero, pos_zero, mk(-3.0)]),
    torch.cat([mk(-5.0), neg_zero, pos_zero, mk(-7.0)]),
]).to("cuda").contiguous()
out = kernel.run(x)
ref = kernel.reference(x.cpu())
mismatch = (out.cpu() != ref.cpu()).nonzero().flatten().tolist()
print(json.dumps({"kernel": out.cpu().tolist(), "ref": ref.tolist(), "mismatch_rows": mismatch}))