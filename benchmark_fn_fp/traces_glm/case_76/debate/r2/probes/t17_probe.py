
import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_149")
import kernel as K

bits = lambda t: (t.cpu().numpy().view(np.uint16).astype(int)).tolist()
z = torch.tensor([0.0, -0.0], dtype=torch.float16).cuda()
oz, rz = K.run(z), K.reference(z)
codes = np.array([c for c in range(0x8000, 0x10000) if (c & 0x7C00) != 0x7C00], dtype=np.uint16)
xn = torch.from_numpy(codes.view(np.int16).copy()).view(torch.float16).cuda().contiguous()
on, rn = K.run(xn), K.reference(xn)
obn = bits(on); rbn = bits(rn)
neg_mism = sum(1 for a, b in zip(obn, rbn) if a != b)
neg_zero_got = sum(1 for a in obn if a == 0)
neg_zero_ref = sum(1 for b in rbn if b == 0)
neg_negzero_got = sum(1 for a in obn if a == 0x8000)
print(json.dumps({"claim": "c3 signed-zero and underflow-to-zero bit patterns",
                  "metric": "exact FP16 output bit patterns: +0=0x0000, -0=0x8000, sigmoid(+-0)=0.5 (0x3800)",
                  "in_0_bits": bits(oz), "ref_0_bits": bits(rz),
                  "n_negative_inputs": len(codes), "negative_mismatch_count": neg_mism,
                  "neg_zero_got": neg_zero_got, "neg_zero_ref": neg_zero_ref,
                  "neg_negzero_got": neg_negzero_got,
                  "verdict_supports_bug": (bits(oz) != bits(rz)) or neg_zero_got != neg_zero_ref or neg_negzero_got != 0 or neg_mism > 0}))
