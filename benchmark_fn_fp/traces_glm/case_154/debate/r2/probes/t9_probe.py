import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
rows = [
 [65504, 1, 65504, 65504],      # a*d = b*c exactly? a*d=65504*65504, b*c=1*65504 -> no; imag = (1*65504 - 65504*65504)/...  wait recompute below
 [1, 65504, 65504, 65504],      # b*c = 65504*65504, a*d = 65504*65504 -> imag true = 0
 [65504, 65504, 65504, 65504],  # b*c - a*d = 0
 [65504, 2048, 65504, 16352],   # b*c = a*d = 65504*65504 approx via powers
 [60000, 60000, 60000, 60000],
 [-65504, 65504, 65504, -65504],
]
num = torch.tensor([[r[0], r[1]] for r in rows], dtype=torch.float16, device="cuda")
den = torch.tensor([[r[2], r[3]] for r in rows], dtype=torch.float16, device="cuda")
out = k.run(num, den).double()
ref = k.reference(num, den)
err = (out - ref).abs()
tol = 1e-7 + 5e-7 * ref.abs()
viol = err > tol
print("num:", num.tolist())
print("den:", den.tolist())
print("out:", out.tolist())
print("ref:", ref.tolist())
print("err:", err.tolist())
print("tol:", tol.tolist())
print(json.dumps({"rows_tested": len(rows), "n_violations": int(viol.sum()),
                  "max_violation_ratio": float((err / tol)[viol].max().item()) if viol.any() else 0.0,
                  "which_violate": [[int(i//2), int(i%2)] for i in torch.nonzero(viol).flatten().tolist()]}))