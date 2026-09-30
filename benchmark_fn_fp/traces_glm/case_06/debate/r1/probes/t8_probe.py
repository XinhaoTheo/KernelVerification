import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_06/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

torch.manual_seed(0)
results = {}
for nchunks, dim in [(8, 16), (64, 32), (256, 8)]:
    new_states = torch.randn(nchunks, dim, device="cuda", dtype=torch.float32)
    dA_cs = torch.randn(nchunks, device="cuda", dtype=torch.float32) * 0.5 - 0.2
    ref = torch.zeros(dim, device="cuda", dtype=torch.float32)
    for c in range(nchunks):
        ref = torch.exp(dA_cs[c]) * ref + new_states[c]
    out = mod.state_passing_lowbit(new_states, dA_cs)
    err = (out - ref).abs()
    rel = (err / ref.abs().clamp_min(1e-12))
    results[f"nchunks{nchunks}_dim{dim}"] = {
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "max_rel_err": float(rel.max()),
        "n_chunks": nchunks, "dim": dim,
        "on_grid_frac": float(((out / 5e-3).round() - out / 5e-3).abs().max() < 1e-3),
    }
print(json.dumps(results))