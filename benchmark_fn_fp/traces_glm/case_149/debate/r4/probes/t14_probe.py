import importlib.util, json, struct
import torch
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_149/kernel.py")
kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)

codes = torch.arange(65536, dtype=torch.int16)
vals = codes.view(torch.float16)
sel = torch.isfinite(vals) & (vals <= -16.0)
x = vals[sel]
print("n_neg_tail_inputs", int(x.numel()))
xc = x.to("cuda")
out = kernel.run(xc)

xd = xc.cpu().to(torch.float64)
ref = torch.tensor([struct.unpack("<e", struct.pack("<e", v))[0] for v in torch.sigmoid(xd).tolist()], dtype=torch.float16).to("cuda")
ob = out.view(torch.int16).cpu(); rb = ref.view(torch.int16).cpu()
mism = (ob != rb).nonzero().flatten()
def is_sub(h): return (int(h) & 0x7C00) == 0 and (int(h) & 0x03FF) != 0
sub_out = sum(1 for h in ob.tolist() if is_sub(h)); sub_ref = sum(1 for h in rb.tolist() if is_sub(h))
neg_zero_out = sum(1 for h in ob.tolist() if (int(h) & 0xFFFF) == 0x8000)
pos_zero_out = sum(1 for h in ob.tolist() if (int(h) & 0xFFFF) == 0)
# mpmath verification of any mismatches
import mpmath
mpmath.mp.dps = 60
details = []
for i in mism.tolist()[:30]:
    xval = xc[i].item()
    v = mpmath.sigmoid(mpmath.mpf(float(xval)))
    t = struct.unpack("<e", struct.pack("<e", float(v)))[0]
    details.append({"x": repr(xval), "kernel_bits": hex(int(ob[i]) & 0xFFFF), "ref_bits": hex(int(rb[i]) & 0xFFFF),
                   "mpmath_bits": hex(struct.unpack("<H", struct.pack("<e", t))[0])})
print(json.dumps({"claim": "c2", "metric": "exact FP16 bit equality, subnormal preservation, zero sign",
  "n_inputs": int(x.numel()), "mismatches": int(mism.numel()),
  "subnormal_out_kernel": sub_out, "subnormal_out_ref": sub_ref,
  "neg_zero_out": neg_zero_out, "pos_zero_out": pos_zero_out, "details": details}))
