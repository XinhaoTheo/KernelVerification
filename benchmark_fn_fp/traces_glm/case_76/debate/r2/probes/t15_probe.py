
import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_149")
import kernel as K

codes = np.array([c for c in range(0x10000) if (c & 0x7C00) != 0x7C00], dtype=np.uint16)
assert len(codes) == 15361
x = torch.from_numpy(codes.view(np.int16).copy()).view(torch.float16).cuda().contiguous()
out = K.run(x)
ref = K.reference(x)
ob = out.cpu().numpy().view(np.uint16).astype(int)
rb = ref.cpu().numpy().view(np.uint16).astype(int)
mism = ob != rb
nm = int(mism.sum())
examples = []
for i in np.nonzero(mism)[0][:10]:
    examples.append({"x_bits": int(codes[i]), "x": float(x[i].item()),
                     "got_bits": int(ob[i]), "ref_bits": int(rb[i])})
print(json.dumps({"claim": "c1 exhaustive sweep over all 15361 finite FP16 encodings",
                  "metric": "exact FP16 bit-pattern equality vs reference() oracle",
                  "n": 15361, "mismatch_count": nm, "examples": examples,
                  "verdict_supports_bug": nm > 0}))
