
import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_150/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

results = {}
fail = False
for N in (64, 128):
    for B in (1, 3, 128):
        x = torch.randint(0, 3329, (B, N), dtype=torch.int32, device="cuda")
        out = k.run(x)
        ref = k.reference(x)
        ok = bool(torch.equal(out, ref))
        results[f"N{N}_B{B}"] = ok
        if not ok: fail = True
# extreme values: all max, all zero, delta, single spike, sequential
tests = {
  "all_3328_N64": torch.full((4,64), 3328, dtype=torch.int32, device="cuda"),
  "all_zero_N128": torch.zeros((5,128), dtype=torch.int32, device="cuda"),
  "delta_N64": torch.eye(64, dtype=torch.int32, device="cuda")[:5]*3328,
  "seq_N128": (torch.arange(128, dtype=torch.int32, device="cuda")*26 % 3329).expand(8,128).contiguous(),
}
for name, x in tests.items():
    out = k.run(x); ref = k.reference(x)
    ok = bool(torch.equal(out, ref)); results[name]=ok
    if not ok: fail = True
# nonzero storage offset contiguous view
big = torch.randint(0,3329,(200,128), dtype=torch.int32, device="cuda")
view = big[70:80, :64].contiguous()  # still contiguous after slice-copy? do real offset view:
base = torch.randint(0,3329,(160,128), dtype=torch.int32, device="cuda")
view = base.narrow(0, 16, 64)  # contiguous view with nonzero storage offset
out = k.run(view); ref = k.reference(view)
results["offset_view_N128_B64"] = bool(torch.equal(out, ref))
if not torch.equal(out, ref): fail = True
# random sweeps
for seed in range(10):
    N = 64 if seed%2 else 128
    x = torch.randint(0,3329,(16,N), dtype=torch.int32, device="cuda")
    if not torch.equal(k.run(x), k.reference(x)):
        fail=True; results[f"seed{seed}"]=False
print(json.dumps({"metric":"exact torch.equal vs reference DFT (tol 0)", "fail":fail, "results":results}))
