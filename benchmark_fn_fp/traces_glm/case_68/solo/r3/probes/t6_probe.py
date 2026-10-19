import torch, sys
sys.path.insert(0, "/root/cases/case_141")
import kernel

torch.manual_seed(0)
vals = torch.tensor([0, 6, 7, 13, 4294967295, 4294967289, 4294967290,
                     2**31, 2**31-1, 2**32-8, 2**32-15, 0xFFFFFFF0], dtype=torch.uint32)
# plus random spread
rand = torch.randint(0, 2**32, (10000,), dtype=torch.int64, generator=torch.Generator().manual_seed(1)).to(torch.uint32)
idx = torch.cat([vals, rand]).cuda()
out = kernel.run(idx)
ref = kernel.reference(idx.cpu()).cuda()
mismatches = (out != ref).any(dim=1)
print("N:", idx.numel(), "mismatches:", mismatches.sum().item())
bad = idx[mismatches][:10].to(torch.int64).tolist()
badout = out[mismatches][:10].to(torch.int64).tolist()
badref = ref[mismatches][:10].to(torch.int64).tolist()
print("bad idx:", bad)
print("kernel:", badout)
print("ref:   ", badref)
import json
print(json.dumps({"mismatches": int(mismatches.sum().item()), "N": int(idx.numel()),
                  "example_idx": bad[:5], "kernel_rows": [o[0] for o in badout[:5]],
                  "ref_rows": [r[0] for r in badref[:5]]}))
