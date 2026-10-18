
import torch, json, sys
sys.path.insert(0, "/root/cases/case_149")
import kernel as K

# all finite FP16 encodings (exclude NaN/Inf: exponent all ones)
codes = [c for c in range(0x10000) if (c & 0x7C00) != 0x7C00]
x = torch.tensor(codes, dtype=torch.int16).view(torch.float16).cuda().contiguous()
assert x.numel() == 15361
out = K.run(x)
ref = K.reference(x)
ob = out.view(torch.int16).cpu().numpy().astype(int)
rb = ref.view(torch.int16).cpu().numpy().astype(int)
mism = (ob != rb)
nm = int(mism.sum())
examples = []
for i in torch.nonzero(torch.from_numpy(mism)).flatten().tolist()[:10]:
    examples.append({"x_bits": codes[i], "x": float(torch.tensor(codes[i], dtype=torch.int16).view(torch.float16)),
                     "got_bits": int(ob[i]) & 0xFFFF, "ref_bits": int(rb[i]) & 0xFFFF})
print(json.dumps({"claim": "c1 exhaustive sweep over all 15361 finite FP16 encodings",
                  "metric": "exact FP16 bit-pattern equality vs reference() oracle (count of mismatching elements)",
                  "n": 15361, "mismatch_count": nm, "examples": examples,
                  "verdict_supports_bug": nm > 0}))
