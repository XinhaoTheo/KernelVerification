import json, sys
sys.path.insert(0, "/root/cases/case_159")
import kernel, torch
# Random {-1,0,1} matrices are statistically noncommuting; N=16 crosses the chunk boundary.
m, b, i = kernel.make_inputs(device="cuda", batch=16, length=16, seed=7)
out = kernel.run(m, b, i)
ref = kernel.reference(m, b, i)
diff = (out != ref)
first8_ok = bool((out[:, :8] == ref[:, :8]).all().item())
last8_ok = bool((out[:, 8:] == ref[:, 8:]).all().item())
# sanity: matrices indeed noncommuting for most rows
M = m.to(torch.int64)
noncomm = sum(1 for r in range(16) for t in range(15) if not torch.equal(M[r, t] @ M[r, t+1], M[r, t+1] @ M[r, t]))
print(json.dumps({"claim": "c3", "noncommuting_pairs_seen": noncomm, "steps_0_7_exact": first8_ok, "steps_8_15_exact": last8_ok, "n_mismatch": int(diff.sum().item())}))