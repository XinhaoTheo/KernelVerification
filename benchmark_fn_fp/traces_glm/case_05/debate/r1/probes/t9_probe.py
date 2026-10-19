import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_06/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

torch.manual_seed(1)
results = {}
# Case A: tiny new_states (magnitude < 2.5e-3)
nchunks, dim = 4, 8
new_states = torch.randn(nchunks, dim, device="cuda", dtype=torch.float32) * 1e-3
dA_cs = -torch.ones(nchunks, device="cuda", dtype=torch.float32) * 0.5
ref = torch.zeros(dim, device="cuda", dtype=torch.float32)
for c in range(nchunks):
    ref = torch.exp(dA_cs[c]) * ref + new_states[c]
out = mod.state_passing_lowbit(new_states, dA_cs)
rel = ((out - ref).abs() / ref.abs().clamp_min(1e-12))
results["tiny_new_states"] = {
    "kernel_final": out.tolist(), "ref_final": ref.tolist(),
    "zeros_in_kernel": int((out == 0).sum()), "zeros_in_ref": int((ref == 0).sum()),
    "max_abs_err": float((out - ref).abs().max()),
    "max_rel_err": float(rel.max()), "ref_max_abs": float(ref.abs().max()),
}
# Case B: strong decay
nchunks, dim = 3, 8
new_states = torch.randn(nchunks, dim, device="cuda", dtype=torch.float32)
dA_cs = -torch.full((nchunks,), 5.0, device="cuda", dtype=torch.float32)
ref = torch.zeros(dim, device="cuda", dtype=torch.float32)
for c in range(nchunks):
    ref = torch.exp(dA_cs[c]) * ref + new_states[c]
out = mod.state_passing_lowbit(new_states, dA_cs)
rel = ((out - ref).abs() / ref.abs().clamp_min(1e-12))
results["strong_decay"] = {
    "kernel_final": out.tolist(), "ref_final": ref.tolist(),
    "max_abs_err": float((out - ref).abs().max()),
    "max_rel_err": float(rel.max()), "ref_max_abs": float(ref.abs().max()),
}
print(json.dumps(results))