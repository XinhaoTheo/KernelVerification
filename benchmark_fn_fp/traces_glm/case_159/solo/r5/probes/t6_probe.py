import torch, sys, json
sys.path.insert(0, "/root/cases/case_159")
import kernel

results = []
def check(B, N, seed, tag):
    g = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8).cuda()
    bias = torch.randint(-32768, 32768, (B, N, 2), generator=g, dtype=torch.int16).cuda()
    initial = torch.randint(-32768, 32768, (B, 2), generator=g, dtype=torch.int16).cuda()
    out = kernel.run(matrix, bias, initial)
    ref = kernel.reference(matrix, bias, initial)
    ok = torch.equal(out, ref)
    results.append({"B": B, "N": N, "seed": seed, "tag": tag, "match": bool(ok)})
    return ok

fails = []
# all N 1..32 with a couple of B values
for N in range(1, 33):
    for B, seed in ((1, N), (7, 1000+N), (64, 5000+N)):
        if not check(B, N, seed, "sweep"):
            fails.append((B, N, seed))
# extreme values
g = torch.Generator(device="cpu").manual_seed(1)
for seed in range(20):
    N = 32
    B = 16
    matrix = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8).cuda()
    bias = torch.full((B, N, 2), 32767, dtype=torch.int16).cuda(); bias[:, :, 0] = -32768
    initial = torch.full((B, 2), -32768, dtype=torch.int16).cuda()
    out = kernel.run(matrix, bias, initial)
    ref = kernel.reference(matrix, bias, initial)
    if not torch.equal(out, ref):
        fails.append((B, N, seed, "extreme"))
        results.append({"B": B, "N": N, "seed": seed, "tag": "extreme", "match": False})
    else:
        results.append({"B": B, "N": N, "seed": seed, "tag": "extreme", "match": True})

n_total = len(results)
n_match = sum(1 for r in results if r["match"])
print(json.dumps({"n_total": n_total, "n_match": n_match, "fails": [str(f) for f in fails]}))