import torch, json, sys, math
sys.path.insert(0, "/root/cases/case_133")
import kernel

dev = "cuda"
# odd-quadrant inputs with |r| near pi/4: x near k*pi/2 + pi/4. FP16 quantizes, so take the closest FP16 values.
vals = []
krange = range(-20000, 20001)
targets = [k*math.pi/2 + math.pi/4 for k in krange if abs(k*math.pi/2 + math.pi/4) <= 65504]
ts = torch.tensor(targets, dtype=torch.float64)
lo = ts * (1 - 2**-11); hi = ts * (1 + 2**-11)
lo16 = lo.to(torch.float16); hi16 = hi.to(torch.float16)
cand = torch.cat([lo16.to(torch.float64), hi16.to(torch.float64)]).unique()
# filter to ones whose reduced r is actually near pi/4
def r_of(x):
    q = math.floor(x*(2/math.pi)+0.5)
    r = x - q*1.5703125 - q*0.00048382679489661923
    return r
cand_list = cand.tolist()
sel = [v for v in cand_list if v != 0 and abs(r_of(v)) > 0.7 and abs(r_of(v)) < 0.79]
# dedupe
sel = sorted(set(sel))
# also moderate-magnitude odd-quadrant examples near pi/2
extra = [math.pi/2 + m*1.0/16.0 for m in range(-64, 65)]
sel += extra
sel = sorted(set(sel))
sel = [v for v in sel if abs(v) <= 65504]

x = torch.tensor(sel[:131072], dtype=torch.float16, device=dev)
out = kernel.run(x)
ref = torch.sin(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
viol = err > 1e-6
res = {
    "tolerance": 1e-6,
    "n": int(x.numel()),
    "n_violations": int(viol.sum().item()),
    "max_err": float(err.max().item()),
}
if viol.any():
    vi = torch.nonzero(viol).flatten()
    res["example_violations"] = [{"x": float(x[i].item()), "err": float(err[i].item()), "actual": float(out[i].item()), "ref": float(ref[i].item())} for i in vi[:10].tolist()]
    res["worst_x"] = float(x[int(err.argmax())].item())
print(json.dumps(res))
