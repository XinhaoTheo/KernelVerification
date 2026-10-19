import json, sys
sys.path.insert(0, "/root/cases/case_159")
import kernel, torch

def noncomm_count(m):
    M = m.to(torch.int64).cpu()
    A = M[:, :-1]; B = M[:, 1:]
    AB1 = A[...,0,0]*B[...,0,0] + A[...,0,1]*B[...,1,0]
    AB2 = A[...,0,0]*B[...,0,1] + A[...,0,1]*B[...,1,1]
    AB3 = A[...,1,0]*B[...,0,0] + A[...,1,1]*B[...,1,0]
    AB4 = A[...,1,0]*B[...,0,1] + A[...,1,1]*B[...,1,1]
    BA1 = B[...,0,0]*A[...,0,0] + B[...,0,1]*A[...,1,0]
    BA2 = B[...,0,0]*A[...,0,1] + B[...,0,1]*A[...,1,1]
    BA3 = B[...,1,0]*A[...,0,0] + B[...,1,1]*A[...,1,0]
    BA4 = B[...,1,0]*A[...,0,1] + B[...,1,1]*A[...,1,1]
    return int((~((AB1==BA1)&(AB2==BA2)&(AB3==BA3)&(AB4==BA4))).sum().item()), int(A.numel()//4)

res = {}
for (Bb, N, seed) in [(16,16,7), (7,32,0), (16,32,7), (16,32,3)]:
    m, b, i = kernel.make_inputs(device="cuda", batch=Bb, length=N, seed=seed)
    try:
        out = kernel.run(m, b, i)
        ref = kernel.reference(m, b, i)
        diff = (out != ref)
        nc, tot = noncomm_count(m)
        nz = torch.nonzero(diff.reshape(-1), as_tuple=False).reshape(-1)
        first_step = int(nz[0].item() % N) if nz.numel() else -1
        res[f"B{Bb}_N{N}_s{seed}"] = {
            "noncommuting_adjacent_pairs": nc, "total_pairs": tot,
            "exact": bool(not diff.any().item()),
            "n_mismatch": int(diff.sum().item()),
            "first_diff_step": first_step,
            "steps_lt8_exact": bool((out[:, :min(8,N)] == ref[:, :min(8,N)]).all().item()),
            "max_abs_err": int((out - ref).abs().max().item()),
        }
    except Exception as e:
        res[f"B{Bb}_N{N}_s{seed}"] = {"error": f"{type(e).__name__}: {e}"}
print(json.dumps({"claim": "c3", "results": res}))