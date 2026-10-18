import json, math, importlib.util
import torch

spec = importlib.util.spec_from_file_location("k114", "/root/cases/case_114/kernel.py")
k114 = importlib.util.module_from_spec(spec); spec.loader.exec_module(k114)

dev = "cuda"
QT, HQ, HK, D, PAGE = 33, 2, 1, 64, 16
length, window, P = 130, 50, 16

g = torch.Generator().manual_seed(7)
q = (2*torch.rand((1, QT, HQ, D), generator=g)-1).half().to(dev)
k = torch.zeros((P, PAGE, HK, D), dtype=torch.float16, device=dev)
v = torch.zeros_like(k)
table = torch.arange(P, dtype=torch.int32, device=dev).reshape(1, P)
lengths = torch.tensor([length], dtype=torch.int32, device=dev)

lo = length - QT - window + 1  # 48
for t in range(lo, length):
    p, s = t // PAGE, t % PAGE
    k[p, s] = (2*torch.rand((HK, D), generator=g)-1).half()
    v[p, s] = (2*torch.rand((HK, D), generator=g)-1).half()
k[:3] = float("nan"); v[:3] = float("nan")

# Public path: run() calls validate_inputs first.
try:
    validation_passed = bool(k114.validate_inputs(q, k, v, table, lengths, window))
except AssertionError as e:
    validation_passed = False
    print("validate_inputs raised:", e)

inputs_before = [t.clone() for t in (q, k, v, table, lengths)]
out = k114.run(q, k, v, table, lengths, window)
torch.cuda.synchronize()
unchanged = all(torch.equal(a, b) for a, b in zip((q, k, v, table, lengths), inputs_before))

finite = bool(torch.isfinite(out).all())
nan_rows = int(torch.isnan(out).any(dim=-1).sum().item())
print(json.dumps({"validation_passed": validation_passed,
                  "public_run_output_finite": finite,
                  "nan_or_inf_rows": nan_rows,
                  "total_rows": out.shape[1]*out.shape[2],
                  "inputs_bit_for_bit_unchanged": bool(unchanged)}))