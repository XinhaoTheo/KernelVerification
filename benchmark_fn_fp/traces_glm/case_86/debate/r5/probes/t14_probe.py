import json, sys
sys.path.insert(0, "/root/cases/case_159")
import kernel, torch

def ref_seq(m, b, i):
    return kernel.reference(m, b, i)

def noncomm_count(m):
    M = m.to(torch.int64).cpu()
    A = M[:, :-1]  # [B,N-1,2,2]
    B = M[:, 1:]
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
for (B, N, seed) in [(16,16,7), (7,32,0), (16,32,7), (16,32,3)]:
    m, b, i = kernel.make_inputs(device="cuda", batch=B, length=N, seed=seed)
    out = kernel.run(m, b, i)
    ref = ref_seq(m, b, i)
    diff = (out != ref)
    nc, tot = noncomm_count(m)
    first_step = int(torch.nonzero(diff, as_tuple=False).view(-1)[0].item() % N) if diff.any() else -1
    res[f"B{B}_N{N}_s{seed}"] = {
        "noncommuting_adjacent_pairs": nc, "total_pairs": tot,
        "exact": bool(not diff.any().item()),
        "n_mismatch": int(diff.sum().item()),
        "first_diff_step": first_step,
        "steps_lt8_exact": bool((out[:, :min(8,N)] == ref[:, :min(8,N)]).all().item()),
    }
print(json.dumps({"claim": "c3", "results": res}))