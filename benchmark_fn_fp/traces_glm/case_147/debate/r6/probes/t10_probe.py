import torch, importlib.util, json, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_147/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def gen_near(v, w=0.06):
    q = torch.linspace(v-w, v+w, 2001, dtype=torch.float32).to(torch.float16)
    q = torch.unique(q)
    return q[(q > 0)]

xs = torch.cat([gen_near(0.66), gen_near(2.414213562373095)]).to("cuda")
o = k.run(xs)
ref = torch.atan(xs.to(torch.float64))
e = (o.to(torch.float64) - ref).abs()
j = int(e.argmax())
b_low = xs <= 0.66
res = {"probe": "branch boundaries 0.66 and 2.414",
 "n": int(xs.numel()), "max_abs_err": float(e.max().item()),
 "tolerance": 1e-6, "exceeds_tol": bool(e.max().item() > 1e-6),
 "argmax_x": float(xs[j].item()),
 "max_err_lower_branch": float(e[b_low].max().item()),
 "max_err_middle_branch": float(e[~b_low].max().item()),
 "n_violations": int((e > 1e-6).sum().item())}
print(json.dumps(res))
