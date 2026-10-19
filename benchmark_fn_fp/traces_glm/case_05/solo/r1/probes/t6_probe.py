
import torch, sys, json
sys.path.insert(0, "/root/cases/case_06")
from kernel import state_passing_lowbit

torch.manual_seed(0)
dev = "cuda"
results = []
for nchunks, dim in [(8, 64), (64, 128), (256, 64)]:
    new_states = torch.randn(nchunks, dim, device=dev, dtype=torch.float32)
    dA_cs = -0.05 * torch.rand(nchunks, device=dev, dtype=torch.float32) - 0.01  # log-decays in (-0.06, -0.01)
    out = state_passing_lowbit(new_states, dA_cs)
    # exact reference in fp64
    st = torch.zeros(dim, device=dev, dtype=torch.float64)
    for c in range(nchunks):
        st = torch.exp(dA_cs[c].double()) * st + new_states[c].double()
    err = (out.double() - st).abs()
    results.append({
        "nchunks": nchunks, "dim": dim,
        "max_abs_err": err.max().item(),
        "mean_abs_err": err.mean().item(),
        "rel_err_max": (err / (st.abs() + 1e-30)).max().item(),
    })
print(json.dumps({"metric": "final_state abs error vs exact fp64 recurrence", "results": results}))
