import torch, sys, json
import triton
import triton.language as tl

@triton.jit
def _gather_probe(IN, OUT, N: tl.constexpr):
    lane = tl.arange(0, N)
    v = tl.load(IN + lane)
    idx = (N - 1) - lane
    w = tl.gather(v, idx, axis=0)
    tl.store(OUT + lane, w)

results = {}
try:
    for N in (64, 128):
        x = torch.arange(N, dtype=torch.int32, device="cuda")
        out = torch.empty_like(x)
        _gather_probe[(1,)](x, out, N=N)
        ok = bool(torch.equal(out, x.flip(0)))
        results[f"gather_reverse_perm_ok_N{N}"] = ok
        if not ok:
            results[f"out_first8_N{N}"] = out[:8].tolist()
            results[f"expected_first8_N{N}"] = x.flip(0)[:8].tolist()
    results["gather_semantics_ok"] = all(v for k, v in results.items() if k.startswith("gather_reverse"))
except Exception as e:
    results = {"error": f"{type(e).__name__}: {e}"}
print(json.dumps(results))