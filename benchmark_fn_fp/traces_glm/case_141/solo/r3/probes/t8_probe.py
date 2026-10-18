import torch, sys, json
sys.path.insert(0, "/root/cases/case_141")
import kernel

vals = torch.tensor([0, 6, 7, 13, 4294967295, 4294967289, 4294967290,
                     2**31, 2**31-1, 2**32-8, 2**32-15, 0xFFFFFFF0], dtype=torch.uint32)
rand = torch.randint(0, 2**32, (10000,), dtype=torch.int64, generator=torch.Generator().manual_seed(1)).to(torch.uint32)
idx_cpu = torch.cat([vals, rand])
idx = idx_cpu.cuda()
out = kernel.run(idx).cpu().to(torch.int64)
ref = kernel.reference(idx_cpu).to(torch.int64)
i64 = idx_cpu.to(torch.int64)
mism = (out != ref).any(dim=1)
res = {
 "N": int(i64.numel()),
 "mismatches": int(mism.sum().item()),
 "example_idx": i64[mism][:8].tolist(),
 "kernel_out": out[mism][:8].tolist(),
 "ref_out": ref[mism][:8].tolist(),
 "cols_out_of_range": int((out[:,1] > 6).sum().item()),
 "invariant_failures": int((7*out[:,0] + out[:,1] != i64).sum().item()),
 "extremes_exact": {str(int(v)): bool((out[i]==ref[i]).all().item()) for i,v in enumerate(i64[:12])},
}
print(json.dumps(res))
