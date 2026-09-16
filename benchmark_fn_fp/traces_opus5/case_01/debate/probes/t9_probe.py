
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_01/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
dev = "cuda"

def ref_idx(target, draft, inv_q):
    prob = torch.clamp(target - draft, min=0.0)
    score = prob * inv_q
    mx = score.max(dim=1, keepdim=True).values
    # first occurrence of max
    return (score == mx).float().argmax(dim=1)

out = {}

# Case A: normalized probability rows, generic (positive residual exists)
B, V = 8, 128
target = torch.softmax(torch.randn(B, V, device=dev), dim=1)
draft  = torch.softmax(torch.randn(B, V, device=dev), dim=1)
q = torch.empty(B, V, device=dev).exponential_(1.0)
inv_q = 1.0 / q
k = m.sample_recovered_tokens(target, draft, inv_q)
r = ref_idx(target, draft, inv_q)
resid = torch.clamp(target - draft, min=0.0)
out["A_generic_mismatch"] = int((k != r).sum())
out["A_generic_zero_mass_selected"] = int((resid.gather(1, k.view(-1,1)).squeeze(1) == 0).sum())
out["A_rows"] = B

# Case B: rows where target elementwise strictly below draft -> no positive residual
B2, V2 = 8, 128
base = torch.softmax(torch.randn(B2, V2, device=dev), dim=1)
draft2 = base
target2 = base * 0.5   # strictly less everywhere (all entries > 0)
q2 = torch.empty(B2, V2, device=dev).exponential_(1.0)
inv_q2 = 1.0 / q2
k2 = m.sample_recovered_tokens(target2, draft2, inv_q2)
r2 = ref_idx(target2, draft2, inv_q2)
resid2 = torch.clamp(target2 - draft2, min=0.0)
sel_resid2 = resid2.gather(1, k2.view(-1,1)).squeeze(1)
out["B_rows"] = B2
out["B_all_residuals_nonpositive"] = bool((target2 - draft2 <= 0).all())
out["B_mismatch_vs_ref"] = int((k2 != r2).sum())
out["B_zero_mass_selected"] = int((sel_resid2 == 0).sum())
out["B_kernel_idx"] = k2.tolist()
out["B_ref_idx"] = r2.tolist()
# unclamped argmax prediction (what kernel should be doing per its source)
unc = ((target2 - draft2) * inv_q2)
out["B_unclamped_argmax"] = unc.argmax(dim=1).tolist()
out["B_kernel_matches_unclamped"] = bool((k2 == unc.argmax(dim=1)).all())

# Case C: only one token has positive residual, rest negative (sharper: correct answer unique)
B3, V3 = 8, 128
draft3 = torch.softmax(torch.randn(B3, V3, device=dev), dim=1)
target3 = draft3 * 0.5
pos = torch.randint(0, V3, (B3,), device=dev)
target3[torch.arange(B3, device=dev), pos] = draft3[torch.arange(B3, device=dev), pos] + 0.01
q3 = torch.empty(B3, V3, device=dev).exponential_(1.0)
inv_q3 = 1.0 / q3
k3 = m.sample_recovered_tokens(target3, draft3, inv_q3)
r3 = ref_idx(target3, draft3, inv_q3)
resid3 = torch.clamp(target3 - draft3, min=0.0)
out["C_rows"] = B3
out["C_unique_positive_per_row"] = int(((resid3 > 0).sum(dim=1) == 1).sum())
out["C_mismatch_vs_ref"] = int((k3 != r3).sum())
out["C_zero_mass_selected"] = int((resid3.gather(1, k3.view(-1,1)).squeeze(1) == 0).sum())
out["C_kernel_idx"] = k3.tolist()
out["C_ref_idx"] = r3.tolist()
print(json.dumps(out))
