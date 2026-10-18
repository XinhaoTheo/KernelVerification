
import torch, numpy as np, json
import importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_137/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

bits = np.arange(0, 65536, dtype=np.uint16)
fp = bits.view(np.float16)
finite = np.isfinite(fp)
vals = torch.from_numpy(fp[finite].copy()).to(torch.float16).cuda()

ref = k.reference(vals)
out = []
for i in range(0, vals.numel(), 65536):
    out.append(k.run(vals[i:i+65536]))
out = torch.cat(out).cpu()
ref = ref.cpu()

err = (out - ref).abs()
res = {"metric":"max abs error over all finite FP16 inputs","n": int(vals.numel()),
 "max_abs_err": float(err.max()), "argmax_bits": int(bits[finite][int(err.argmax())]),
 "argmax_x": float(vals[int(err.argmax())].item()), "tol": 1e-6,
 "num_violations": int((err > 1e-6).sum()), "kernel_allclose_1e-6": bool((err <= 1e-6).all())}
print(json.dumps(res))
