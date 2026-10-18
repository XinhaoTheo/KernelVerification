import torch, json, sys, itertools, random
sys.path.insert(0, "/root/cases/case_154")
import kernel
dev = "cuda"
random.seed(0)
vals = [0.0, -0.0, 65504.0, -65504.0, 6.104e-05, -6.104e-05, 6.0e-08, -6.0e-08, 1.0, -1.0, 5.0e-08, 5.0e-05]
rows = []
for a in vals:
    for b in vals:
        for c in vals:
            for d in vals:
                if c == 0 and d == 0: continue
                rows.append((a, b, c, d))
random.shuffle(rows)
rows = rows[:50000]
num = torch.tensor(rows, dtype=torch.float16, device=dev)
den = torch.tensor([(r[2], r[3]) for r in rows], dtype=torch.float16, device=dev)
out = kernel.run(num, den)
ref = kernel.reference(num.cpu(), den.cpu()).to(dev).double()
ratio = (out.double() - ref).abs() / (1e-7 + 5e-7 * ref.abs())
res = {
  "n": len(rows),
  "max_ratio": float(ratio.max()),
  "violations": int((ratio > 1.0).sum()),
  "output_finite": bool(torch.isfinite(out).all()),
}
# inputs preserved check
num2, den2 = kernel.make_inputs(dev, size=1000, seed=1)
numc, denc = num2.clone(), den2.clone()
kernel.run(num2, den2)
res["inputs_preserved"] = bool(torch.equal(num2, numc) and torch.equal(den2, denc))
if res["violations"] > 0:
    i = int(ratio.flatten().argmax()); r, cc = divmod(i, 2)
    res["worst_row"] = [float(num[r,0]), float(num[r,1]), float(den[r,0]), float(den[r,1]), cc]
    res["worst_actual"] = float(out[r,cc]); res["worst_target"] = float(ref[r,cc])
print(json.dumps(res))
