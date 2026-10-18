
import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_139/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

bits = torch.arange(0, 65536, dtype=torch.int32)
allv = bits.to(torch.uint16).view(torch.float16)
allf = allv[torch.isfinite(allv)].clone()
print("count finite FP16:", allf.numel())

out = k.run(allf.cuda())
ref = k.reference(allf).to(out.device)  # FP64
err = (out.double() - ref).abs()
tol = 1e-8 + 1e-5*ref.abs()
bad = err > tol
print("max abs err:", err.max().item())
print("num violating:", bad.sum().item())
if bad.any():
    idx = bad.nonzero().flatten()
    xs = allf[idx.cpu()][:10].tolist()
    print("worst offenders x:", xs)
    print("err:", err[idx][:10].tolist())
    print("tol:", tol[idx][:10].tolist())
print(json.dumps({"num_violating": int(bad.sum().item()), "max_abs_err": float(err.max().item())}))
