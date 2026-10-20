import torch, sys, json
sys.path.insert(0, '/root/cases/case_168')
from kernel import run, reference

results = {}
for N in [1, 2, 255, 256, 257, 1027, 131072]:
    x = torch.randint(0, 256, (N,), dtype=torch.uint8, device='cuda')
    out = run(x)
    ref = reference(x)
    eq = torch.equal(out, ref)
    # tail indices specifically (last partial block region)
    tail_ok = True
    if N % 256 != 0:
        start = (N // 256) * 256
        tail_ok = torch.equal(out[start:], ref[start:])
    results[str(N)] = {"all_exact": bool(eq), "tail_exact": bool(tail_ok), "shape": tuple(out.shape), "dtype": str(out.dtype)}
print(json.dumps({"metric": "exact equality at every position for edge lengths incl. partial final block tails",
                  "results": results}))