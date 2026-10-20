import numpy as np, torch, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/numerical_cases/case_q/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
q, c, e = k.make_inputs("cuda")
out = k.run(q, c, e)
d = ((c.double() - q.double())**2).sum(dim=1)
ref_idx = int(np.argmin(d.cpu().numpy()))  # argmin picks smallest index on ties
ref = e[ref_idx]
rel = (out - ref).norm().item() / max(ref.norm().item(), 1e-12)
# kernel's winner for diagnosis
qd = np.floor(q.double().cpu().numpy()*8+0.5)/8
cd = np.floor(c.double().cpu().numpy()*8+0.5)/8
kd = ((cd - qd)**2).sum(1)
kidx = int(np.argmin(kd))
print({"rel_err": rel, "ref_idx": ref_idx, "kernel_idx": kidx,
       "true_d": d.cpu().numpy().tolist(), "quant_d": kd.tolist(),
       "output": out.cpu().numpy().tolist(), "ref": ref.cpu().numpy().tolist()})
