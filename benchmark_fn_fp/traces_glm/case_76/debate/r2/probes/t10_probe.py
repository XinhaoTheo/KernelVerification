
import torch, json, sys
sys.path.insert(0, "/root/cases/case_149")
import kernel as K

# negative inputs only, covering subnormal-output range ~[-24,-8] and beyond to -65504
codes = [c for c in range(0x10000) if (c & 0x7C00) != 0x7C00]
neg = torch.tensor(codes, dtype=torch.int16).view(torch.float16)
neg = neg[neg < 0]
x = neg.cuda().contiguous()
out = K.run(x)
ref = K.reference(x)
ob = out.view(torch.int16).cpu().numpy().astype(int) & 0xFFFF
rb = ref.view(torch.int16).cpu().numpy().astype(int) & 0xFFFF
xb = x.view(torch.int16).cpu().numpy().astype(int) & 0xFFFF
mism = (ob != rb)
nm = int(mism.sum())
sub_mask = ((rb != 0) & ((rb & 0x7C00) == 0))  # reference output is positive FP16 subnormal
n_sub = int(sub_mask.sum())
sub_mism = int((mism & sub_mask).sum())
examples = []
idxs = torch.nonzero(torch.from_numpy(mism & sub_mask)).flatten().tolist()[:10]
for i in idxs:
    examples.append({"x_bits": xb[i], "got_bits": ob[i], "ref_bits": rb[i]})
# also count outputs that reference maps to +0 vs kernel maps to +0
zero_ref = int((rb == 0).sum()); zero_got = int((ob == 0).sum())
print(json.dumps({"claim": "c2 subnormal-output focus (all negative finite FP16 inputs)",
                  "metric": "exact FP16 bit-pattern equality vs reference for subnormal-reference outputs; FP16 subnormal step = 2^-24",
                  "n_negative_inputs": int(x.numel()), "mismatch_count_all": nm,
                  "n_ref_subnormal_outputs": n_sub, "mismatch_subnormal": sub_mism,
                  "examples": examples, "zero_ref_count": zero_ref, "zero_got_count": zero_got,
                  "verdict_supports_bug": sub_mism > 0}))
