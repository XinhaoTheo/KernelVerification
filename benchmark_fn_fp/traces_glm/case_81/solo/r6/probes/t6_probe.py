
import torch, json, sys
sys.path.insert(0, "/root/cases/case_154")
import kernel

dev = "cuda"
sub = torch.tensor(5.96e-8, dtype=torch.float16)  # smallest fp16 subnormal

# Case family: c=d=subnormal, numerator chosen so a*c+b*d nearly cancels.
# a=1, b=1024 -> products ~6e-8 each, opposite signs -> real ~ 0; imag similar.
num_rows = [
    (1.0, 1024.0),
    (1.0, -1024.0),
    (1024.0, 1.0),
    (1.0, 1023.0),
    (1.0, 1025.0),
    (3.0, 3072.0),
    (65504.0, 65504.0),   # no cancellation: imag cancels exactly? b*c - a*d with a=b -> 0
]
rows = []
for (a, b) in num_rows:
    for (c, d) in [(sub, sub), (sub, -sub), (-sub, sub), (sub, torch.tensor(2.0**-13, dtype=torch.float16))]:
        rows.append((a, b, c, d))

numerator = torch.tensor([[r[0], r[1]] for r in rows], dtype=torch.float16, device=dev)
denominator = torch.tensor([[r[2], r[3]] for r in rows], dtype=torch.float16, device=dev)

out = kernel.run(numerator, denominator)
ref = kernel.reference(numerator, denominator).to(torch.float32)

tol = 1e-7 + 5e-7 * ref.abs()
err = (out.to(torch.float64) - ref.to(torch.float64)).abs()
ok = err <= tol
print("per-row:")
for i in range(len(rows)):
    print(i, "in=", rows[i], "out=", out[i].tolist(), "ref=", ref[i].tolist(),
          "err=", err[i].tolist(), "tol=", tol[i].tolist(), "pass=", ok[i].tolist())

viol = (~ok).sum().item()
print(json.dumps({
    "metric": "componentwise |actual-target| vs 1e-7+5e-7|target|",
    "rows": len(rows),
    "violating_components": viol,
    "max_abs_err": float(err.max()),
    "max_violation_ratio": float((err / tol).max()),
    "any_violation": bool(viol > 0),
}))
