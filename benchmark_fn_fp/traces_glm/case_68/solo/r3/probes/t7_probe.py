import torch, sys, json
sys.path.insert(0, "/root/cases/case_141")
import kernel

vals = torch.tensor([0, 6, 7, 13, 4294967295, 4294967289, 4294967290,
                     2**31, 2**31-1, 2**32-8, 2**32-15, 0xFFFFFFF0], dtype=torch.uint32)
rand = torch.randint(0, 2**32, (10000,), dtype=torch.int64, generator=torch.Generator().manual_seed(1)).to(torch.uint32)
idx = torch.cat([vals, rand]).cuda()
out = kernel.run(idx)
ref = kernel.reference(idx.cpu()).cuda()
mism = (out != ref).any(dim=1)
idx64 = idx.to(torch.int64)
bad_idx = idx64[mism][:8].tolist()
bad_out = out[mism][:8].to(torch.int64).tolist()
bad_ref = ref[mism][:8].to(torch.int64).tolist()
# invariant check
bad_col = (out[:,1].to(torch.int64) > 6).sum().item()
sum_ok = ((7*out[:,0].to(torch.int64) + out[:,1].to(torch.int64)) == idx64).sum().item()
print(json.dumps({"N": int(idx.numel()), "mismatches": int(mism.sum().item()),
 "example_idx": bad_idx, "kernel_out": bad_out, "ref_out": bad_ref,
 "cols_out_of_range": int(bad_col), "invariant_7row_plus_col_failures": int(idx.numel()-sum_ok)}))
