import sys, json, torch
# Exhaustively compare the kernel's arithmetic path (fp32 multiply + one fp16 cast,
# i.e. float32(int)*scale then .half()) against the reference path (double product then
# .half()) for every quant 0..15, zero 1..16, and every fp16 scale value in (0, 1/32].
quant = torch.arange(16, dtype=torch.float32)
zero = torch.arange(1, 17, dtype=torch.float32)
# all positive fp16 values <= 1/32 (normals and subnormals)
bits = torch.arange(1, 0x4000, dtype=torch.int32)  # subnormals: exponent 0
sub = (bits.view(torch.int16).to(torch.float32))
bits2 = torch.arange(0x0400, 0x3801, dtype=torch.int32)
norm = (bits2.view(torch.int16).to(torch.float32))
scales = torch.cat([sub, norm]).to(torch.float32)
scales = scales[(scales > 0) & (scales <= 0.03125)]
sc = scales.to(torch.float64).to(torch.float16).to(torch.float32)  # exact fp16 values
Q = quant[:, None, None]  # 16
Z = zero[None, :, None]   # 16
S = sc[None, None, :]     # S
# kernel path: (q-z) as int->fp32 (exact), times fp16-exact scale in fp32, one .half()
kern = ((Q - Z).to(torch.float32) * S).to(torch.float16)
# reference path: exact double product then one .half()
ref = ((Q - Z).to(torch.float64) * S.to(torch.float64)).to(torch.float16)
mismatch = (kern.view(torch.int16) != ref.view(torch.int16))
print(json.dumps({
  "claim": "c3 decoded-weight H16 rounding",
  "n_scales_tested": int(scales.numel()),
  "n_combinations": int(16*16*scales.numel()),
  "n_subnormal_scales": int((scales.view(torch.int16) < 0x0400).sum().item()),
  "n_mismatches": int(mismatch.sum().item()),
  "verdict_bit_identical": bool(not mismatch.any().item())}))