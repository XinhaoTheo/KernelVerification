
import torch, sys
sys.path.insert(0, '/root/cases/case_147')
import kernel

# All finite FP16 values (65536 total, within N<=131072)
bits = torch.arange(65536, dtype=torch.int32)
x_all = bits.to(torch.uint16).view(torch.float16)
finite_mask = torch.isfinite(x_all)
x_fin = x_all[finite_mask]
print("finite count:", x_fin.numel())

x_gpu = x_fin.cuda()
out = kernel.run(x_gpu)
ref = kernel.reference(x_gpu)
err = (out.double() - ref.double()).abs()
mx = err.max().item()
idx = int(err.argmax())
print("max abs err:", mx)
print("worst input:", float(x_fin[idx]), "kernel:", float(out[idx]), "ref:", float(ref[idx]))
print("num over 1e-6:", int((err > 1e-6).sum()))
print("num over 5e-7:", int((err > 5e-7).sum()))
import json
print(json.dumps({"max_abs_err": mx, "worst_x": float(x_fin[idx]), "count_over_tol": int((err > 1e-6).sum()), "n": int(x_fin.numel())}))
