import torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_135/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

# full finite FP16 domain
vals = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.float16)
vals = vals[torch.isfinite(vals)]
x = vals.to("cuda")
out = k.run(x)
ref = k.reference(x)
err = (out.double() - ref.double()).abs()
i = int(err.argmax())
print("N =", x.numel())
print("max abs err =", float(err.max()))
print("worst input =", float(x[i]), "actual =", float(out[i]), "target =", float(ref[i]))
print("count > 1e-6:", int((err > 1e-6).sum()))
# also check output dtype/finite
print("dtype", out.dtype, "finite", bool(torch.isfinite(out).all().item()))
import json
print(json.dumps({"max_abs_err": float(err.max()), "violations": int((err>1e-6).sum()), "n": x.numel()}))