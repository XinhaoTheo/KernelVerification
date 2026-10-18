
import torch, struct, sys
sys.path.insert(0, "/root/cases/case_149")
import kernel as k

# all finite FP16 bit patterns
codes = torch.arange(65536, dtype=torch.int32)
bits = codes.to(torch.uint16) if hasattr(torch, 'uint16') else codes
h = torch.tensor(codes.numpy().view('uint16'), dtype=torch.float16) if False else None
import numpy as np
arr = np.arange(65536, dtype=np.uint16)
x_all = torch.from_numpy(arr.view(np.float16).copy())
finite_mask = torch.isfinite(x_all)
xf = x_all[finite_mask].to("cuda")

out = k.run(xf)
ref = k.reference(xf)

mismatch = (out.view(torch.int16) != ref.view(torch.int16))
nbad = int(mismatch.sum())
res = {"num_inputs": int(xf.numel()), "mismatches": nbad,
       "metric": "exact FP16 bit-pattern comparison vs once-rounded real sigmoid (CPU FP64 reference)"}
if nbad:
    idxs = mismatch.nonzero().flatten()[:20]
    bad_in = xf[idxs].float().cpu().tolist()
    bad_out = out[idxs].cpu().view(torch.int16).tolist()
    bad_ref = ref[idxs].cpu().view(torch.int16).tolist()
    res["examples"] = list(zip(bad_in, bad_out, bad_ref))
    res["max_abs_err"] = float((out.float()-ref.float()).abs().max())
import json
print(json.dumps(res))
