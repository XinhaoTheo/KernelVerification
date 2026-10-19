import sys, json
sys.path.insert(0, "/root/cases/case_159")
import torch
import kernel

results = {}
mismatches = []
total_cases = 0
for N in range(1, 33):
    bad = 0
    for seed in range(4):
        for B in (1, 3, 7, 64):
            m, b, i = kernel.make_inputs(device="cuda", batch=B, length=N, seed=seed)
            out = kernel.run(m, b, i)
            ref = kernel.reference(m.cpu(), b.cpu(), i.cpu()).to(out.device)
            total_cases += 1
            if not torch.equal(out, ref):
                bad += 1
                if len(mismatches) < 5:
                    mismatches.append({"N": N, "B": B, "seed": seed,
                                       "diff_count": int((out != ref).sum()),
                                       "max_abs": int((out - ref).abs().max())})
    results[N] = bad

# also adversarial: extreme bias/initial values, all-singular matrices
m = torch.zeros(1, 32, 2, 2, dtype=torch.int8, device="cuda")
bi = torch.full((1, 32, 2), -32768, dtype=torch.int16, device="cuda")
ini = torch.full((1, 2), -32768, dtype=torch.int16, device="cuda")
out = kernel.run(m, bi, ini)
ref = kernel.reference(m.cpu(), bi.cpu(), ini.cpu()).to(out.device)
edge_zero_ok = bool(torch.equal(out, ref))
total_cases += 1

m = torch.ones(1, 32, 2, 2, dtype=torch.int8, device="cuda")
bi = torch.full((1, 32, 2), 32767, dtype=torch.int16, device="cuda")
ini = torch.full((1, 2), 32767, dtype=torch.int16, device="cuda")
out = kernel.run(m, bi, ini)
ref = kernel.reference(m.cpu(), bi.cpu(), ini.cpu()).to(out.device)
edge_one_ok = bool(torch.equal(out, ref))
total_cases += 1

print(json.dumps({
    "metric": "exact equality (torch.equal) run() vs reference()",
    "total_cases": total_cases,
    "per_N_fail_counts": results,
    "total_failures": sum(results.values()),
    "mismatch_examples": mismatches,
    "edge_zero_matrix_min_bias_ok": edge_zero_ok,
    "edge_one_matrix_max_bias_ok": edge_one_ok,
}))
