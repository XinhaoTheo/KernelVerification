
import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_149")
import kernel as K

codes = np.array([c for c in range(0x10000) if (c & 0x7C00) != 0x7C00], dtype=np.uint16)
allx = torch.from_numpy(codes.view(np.int16).copy()).view(torch.float16)
x = allx[allx < 0].cuda().contiguous()
out = K.run(x)
ref = K.reference(x)
ob = out.cpu().numpy().view(np.uint16).astype(int)
rb = ref.cpu().numpy().view(np.uint16).astype(int)
xb = x.cpu().numpy().view(np.uint16).astype(int)
mism = ob != rb
nm = int(mism.sum())
sub_mask = (rb != 0) & ((rb & 0x7C00) == 0)
n_sub = int(sub_mask.sum())
sub_mism = int((mism & sub_mask).sum())
examples = []
for i in np.nonzero(mism & sub_mask)[0][:10]:
    examples.append({"x_bits": int(xb[i]), "x": float(x[i].item()),
                     "got_bits": int(ob[i]), "ref_bits": int(rb[i])})
print(json.dumps({"claim": "c2 subnormal-output focus (all negative finite FP16 inputs)",
                  "metric": "exact FP16 bit equality vs reference for subnormal-reference outputs (step 2^-24)",
                  "n_negative_inputs": int(x.numel()), "mismatch_count_all": nm,
                  "n_ref_subnormal_outputs": n_sub, "mismatch_subnormal": sub_mism,
                  "examples": examples,
                  "zero_ref_count": int((rb == 0).sum()), "zero_got_count": int((ob == 0).sum()),
                  "verdict_supports_bug": sub_mism > 0}))
