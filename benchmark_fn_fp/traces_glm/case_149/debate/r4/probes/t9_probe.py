import importlib.util, json, struct
import torch
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_149/kernel.py")
kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)

# All finite FP16 encodings
codes = [i for i in range(65536)]
vals = torch.tensor(codes, dtype=torch.uint16).view(torch.float16)
finite = torch.isfinite(vals)
xf = vals[finite]
print("num_finite_fp16_inputs", xf.numel())

# run in one call (65504 <= 131072)
xc = xf.to("cuda")
out = kernel.run(xc)

# reference: FP64 sigmoid -> RNE binary16 via struct '<e'
xd = xc.cpu().to(torch.float64)
ref = torch.tensor([struct.unpack("<e", struct.pack("<e", v))[0] for v in torch.sigmoid(xd).tolist()], dtype=torch.float16).to("cuda")

ob = out.view(torch.int16).cpu()
rb = ref.view(torch.int16).cpu()
mism = (ob != rb).nonzero().flatten()
print("mismatch_count", mism.numel())

# mpmath verification of true single-rounding for mismatches
import mpmath
mpmath.mp.dps = 60
def true_fp16(x):
    v = mpmath.sigmoid(mpmath.mpf(float(x)))
    packed = struct.pack("<e", float(v))
    return struct.unpack("<e", packed)[0]
details = []
for idx in mism.tolist()[:50]:
    xval = xc[idx].item()
    t = true_fp16(xval)
    details.append({"x": repr(xval), "kernel_bits": int(ob[idx].item()) & 0xFFFF,
                    "ref_bits": int(rb[idx].item()) & 0xFFFF,
                    "mpmath_bits": struct.unpack("<H", struct.pack("<e", t))[0]})
real_fail = sum(1 for d in details if d["kernel_bits"] != d["mpmath_bits"])
print(json.dumps({"claim": "c1", "metric": "exact FP16 bit equality vs mpmath single-rounding",
  "n_inputs": xf.numel(), "mismatch_vs_fp64ref": int(mism.numel()),
  "mpmath_verified_kernel_wrong": real_fail, "details": details}))
