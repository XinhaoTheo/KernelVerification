import sys, json, torch
sys.path.insert(0, "/root/cases/case_111")
import kernel as K

cases = []
for m in [1, 2, 17, 33, 127, 511, 512, 513, 1000, 4096]:
    for n in [16, 17, 31, 64, 100, 128, 129, 256, 257, 300, 512]:
        cases.append((m, n))

worst = {"dx": 0.0, "dw": 0.0}
bad = []
for i,(m,n) in enumerate(cases):
    x, w, dy, rstd = K.make_inputs("cuda", m, n, seed=i)
    dx, dw = K.run(x, w, dy, rstd)
    r = K.error_ratios((dx, dw), (x, w, dy, rstd))
    worst["dx"] = max(worst["dx"], r["dx"]); worst["dw"] = max(worst["dw"], r["dw"])
    if r["dx"] > 1 or r["dw"] > 1:
        bad.append(((m,n), r))
    if not torch.isfinite(dx).all() or not torch.isfinite(dw).all():
        bad.append(((m,n), "nonfinite"))
print(json.dumps({"num_cases": len(cases), "worst": worst, "failures": bad[:10]}))