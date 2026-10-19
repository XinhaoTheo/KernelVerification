
import sys, json
sys.path.insert(0, "/root/cases/case_149")
import torch, kernel

# negative FP16 inputs in the subnormal-output region -18.5 < x <= -11.0
patterns = [i for i in range(65536) if not (i >> 10) == 0x1F]
xall = torch.tensor(patterns, dtype=torch.int32).view(torch.float16)
sel = (xall < -11.0) & (xall > -18.5)
x = xall[sel].to("cuda")
out = kernel.run(x)
ref = kernel.reference(x)
out_bits = out.view(torch.int16); ref_bits = ref.view(torch.int16)
mism = (out_bits != ref_bits).nonzero().flatten()
# check whether outputs that should be nonzero subnormals were flushed to zero
should_subnormal = (ref.view(torch.float16).abs() > 0) & (ref_bits & 0x7FFF != 0)
n = int(x.numel())
res = {"n_tested": n, "n_mismatch": int(mism.numel()),
       "flushed_to_zero": int(((out_bits & 0x7FFF) == 0).sum() - ((ref_bits & 0x7FFF) == 0).sum())}
fail = []
for idx in mism[:10].tolist():
    fail.append({"x": x[idx].item(), "got": out[idx].item(),
                 "got_bits": hex(out_bits[idx].item()),
                 "want_bits": hex(ref_bits[idx].item())})
res["first_failures"] = fail
print(json.dumps(res))
