
import torch, json, sys
sys.path.insert(0, "/root/cases/case_02")
from kernel import state_passing

torch.manual_seed(0)
dev = "cuda"
results = []
for nchunks, dim in [(1,1),(4,8),(16,64),(33,128),(128,256)]:
    for decay_scale in [0.1, 1.0, 3.0]:
        new_states = torch.randn(nchunks, dim, device=dev) * (dim**0.5)
        dA_cs = -torch.rand(nchunks, device=dev) * decay_scale
        ref = torch.zeros(dim, device=dev)
        for c in range(nchunks):
            ref = torch.exp(dA_cs[c]) * ref + new_states[c]
        out = state_passing(new_states, dA_cs)
        diff = (out - ref).abs()
        rel = (diff / (ref.abs() + 1e-6)).max().item()
        results.append(dict(nchunks=nchunks, dim=dim, decay_scale=decay_scale,
                            max_abs=diff.max().item(), max_rel=rel,
                            dtype=str(out.dtype), shape=tuple(out.shape),
                            nan=bool(torch.isnan(out).any())))
print(json.dumps(results))
report = dict(metric="max abs & rel error vs sequential fp32 reference",
              all_pass=all(r["max_abs"] < 1e-4 and not r["nan"] for r in results),
              results=results)
print(json.dumps(report))
