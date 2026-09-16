
import json, importlib.util, torch

spec = importlib.util.spec_from_file_location("k", "/root/cases/case_21/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
S, MB, PS, HD = 4, 8, 16, 64
NP = S * MB           # 32 pages -> identity mapping stays in bounds
kv = torch.randn(NP, PS, HD, device='cuda', dtype=torch.float32)

def ref_gather(kv, bt, seq_len):
    t = torch.arange(seq_len, device=kv.device)
    pages = bt[:, t // PS].long()          # [S, T]
    within = (t % PS).view(1, -1).expand(pages.shape)
    return kv[pages, within]

res = {}
for name, bt in [
    ("identity", torch.arange(NP, device='cuda', dtype=torch.int32).reshape(S, MB)),
    ("permutation", torch.randperm(NP, device='cuda').to(torch.int32).reshape(S, MB)),
]:
    for seq_len in (128, 100):   # full and partial-final-block
        out = m.paged_gather(kv, bt, seq_len)
        torch.cuda.synchronize()
        ref = ref_gather(kv, bt, seq_len)
        eq = torch.eq(out, ref)
        tok_eq = eq.all(dim=-1)
        res[f"{name}_len{seq_len}"] = {
            "shape_ok": list(out.shape) == [S, seq_len, HD],
            "exact_match_all": bool(eq.all().item()),
            "mismatched_tokens": int((~tok_eq).sum().item()),
            "total_tokens": int(tok_eq.numel()),
            "frac_tokens_wrong": round(float((~tok_eq).float().mean().item()), 4),
            "max_abs_err": float((out - ref).abs().max().item()),
        }

res["block_table_permutation_sample"] = torch.randperm(0).tolist()  # placeholder
print(json.dumps(res))
