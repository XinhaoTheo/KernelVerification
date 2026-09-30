import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_13/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
kv = torch.randn(2, 64, device="cuda")
out = m.gqa_gather(kv, 8)
torch.cuda.synchronize()
n_rep = 8 // 2
ref = torch.stack([kv[q // n_rep] for q in range(8)])
print(json.dumps({
    "num_q_heads": 8, "num_kv_heads": 2, "n_rep": n_rep, "dim": 64,
    "exact_match_consecutive_contract": bool(torch.equal(out, ref)),
    "mismatched_rows": [q for q in range(8) if not torch.equal(out[q], ref[q])],
    "kernel_kv_head_for_q": [int((q % 2)) for q in range(8)],
    "required_kv_head_for_q": [q // n_rep for q in range(8)],
}))