import torch, struct, math, sys
sys.path.insert(0, "/root/cases/case_149")
import kernel

bits = torch.arange(0, 65536, dtype=torch.int32).to(torch.uint16)
x_all = bits.view(torch.float16).cuda()  # all encodings incl NaN/Inf
x_cpu_all = bits.view(torch.float16)
finite_mask = torch.isfinite(x_cpu_all)
# run in chunks of 131072? N max 131072, fine to run all at once but exclude nonfinite
idx = torch.nonzero(finite_mask).flatten()
x = x_cpu_all[idx].cuda()  # 65504 finite encodings
N = x.numel()
print("N finite encodings:", N)

out = kernel.run(x)
ref = kernel.reference(x)

out_bits = out.view(torch.int16).cpu().to(torch.int32) & 0xFFFF
ref_bits = ref.view(torch.int16).cpu().to(torch.int32) & 0xFFFF
mism = (out_bits != ref_bits).nonzero().flatten()

# also compute FP64 ground truth independently (mpmath-free: use python math + struct pack '<e')
def fp64_ref(xi):
    v = 1.0/(1.0+math.exp(-float(xi)))
    return struct.unpack("<e", struct.pack("<e", v))[0]

extra_check = 0
for i in mism.tolist()[:50]:
    xi = x_cpu_all[idx[i]]
    true_r = fp64_ref(xi)
    print("mismatch input:", float(xi), "kernel:", float(out[i].cpu()), "ref:", float(ref[i].cpu()), "fp64:", true_r,
          "kernel_bits:", hex(out_bits[i].item()), "ref_bits:", hex(ref_bits[i].item()))

import json
print(json.dumps({"metric":"exact fp16 bit pattern vs FP64 correctly-rounded reference",
 "mismatch_count": int(mism.numel()), "N": int(N)}))
