import sys, json, numpy as np, torch
sys.path.insert(0, "/root/evidence_cases/case_e12")
import kernel as k
smoke, W, b, c = k.make_inputs_numpy()
vx = np.array([-1,1,-1,1,1,-1,1,-1,1,1,1,1], dtype=np.float32)
X = vx[None, :]
tX = torch.from_numpy(X).to("cuda"); tW = torch.from_numpy(W).to("cuda")
tb = torch.from_numpy(b).to("cuda"); tc = torch.from_numpy(c).to("cuda")
actual = k.run(tX, tW, tb, tc).double().cpu().numpy()
Wd, bd, cd = W.astype(np.float64), b.astype(np.float64), c.astype(np.float64)
target = 0.25*float(vx[0]) + 0.5*float(vx[1]) + float(np.sum(cd*np.maximum(Wd@vx.astype(np.float64)-bd,0.0)))
err = abs(float(actual[0]) - target)
print(json.dumps({"input": "worst box vertex [-1,1,-1,1,1,-1,1,-1,1,1,1,1]",
 "kernel_output": float(actual[0]), "float64_target": target,
 "absolute_error": err, "tolerance": 1.0, "exceeds_tolerance": bool(err > 1.0),
 "output_finite": bool(np.isfinite(actual).all()), "metric": "absolute error vs float64 reference at the worst vertex, per contract tolerance 1.0"}))