
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_06/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
res = {}
for nchunks, dim in [(8, 64), (32, 128), (128, 64)]:
    new_states = torch.randn(nchunks, dim, device='cuda', dtype=torch.float32) * 0.1
    dA_cs = -torch.rand(nchunks, device='cuda', dtype=torch.float32)  # log-decays <= 0
    out = k.state_passing_lowbit(new_states, dA_cs)
    # exact reference in float64
    st = torch.zeros(dim, device='cuda', dtype=torch.float64)
    ns = new_states.double(); da = dA_cs.double()
    for c in range(nchunks):
        st = torch.exp(da[c]) * st + ns[c]
    ref = st
    err = (out.double() - ref).abs()
    denom = ref.abs().clamp_min(1e-12)
    # check grid alignment of kernel output
    grid = 5e-3
    frac = (out.double()/grid)
    on_grid = (frac - frac.round()).abs().max().item()
    res[f"{nchunks}x{dim}"] = dict(
        max_abs_err=err.max().item(),
        max_rel_err=(err/denom).max().item(),
        ref_absmax=ref.abs().max().item(),
        kernel_output_max_offgrid_frac=on_grid,
        n_mismatch_1e5=(err > 1e-5).sum().item(),
        total=dim,
    )
print(json.dumps(res, indent=2))
