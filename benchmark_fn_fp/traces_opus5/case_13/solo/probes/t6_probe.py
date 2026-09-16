
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_13/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

res = {}
for (nkv, nq) in [(2,8),(4,8),(3,6),(1,4),(8,8)]:
    dim = 4
    kv = torch.arange(nkv*dim, device='cuda', dtype=torch.float32).reshape(nkv, dim)
    out = m.gqa_gather(kv, nq)
    n_rep = nq // nkv
    ref_idx = torch.arange(nq, device='cuda') // n_rep
    ref = kv[ref_idx]
    got_idx = (out[:,0] / dim).to(torch.int64).tolist()
    res[f"nkv{nkv}_nq{nq}"] = {
        "kernel_kv_row_per_qhead": got_idx,
        "contract_kv_row_per_qhead": ref_idx.tolist(),
        "match": bool(torch.equal(out, ref)),
        "max_abs_err": float((out-ref).abs().max()),
    }
print(json.dumps({"metric":"exact row-index mapping vs contract consecutive grouping","results":res}, indent=1))
