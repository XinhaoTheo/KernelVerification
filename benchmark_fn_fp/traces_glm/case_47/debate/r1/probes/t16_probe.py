import json, torch

# Corrected exhaustive comparison: kernel path (fp32 multiply + one fp16 cast)
# vs reference path (exact double product then one .half()) for quant 0..15,
# zero 1..16, and every positive fp16 scale value in (0, 1/32].
qz = (torch.arange(16, dtype=torch.float32)[:, None, None]
      - torch.arange(1, 17, dtype=torch.float32)[None, :, None])  # (16,16,1) q-z, exact in fp32
bits = torch.arange(1, 0x7C00, dtype=torch.int32)
u = torch.arange(1, 0x7C00, dtype=torch.int16).view(torch.uint16)  # unused; build via bitcast below
# build all positive finite fp16 values safely via uint16 reinterpretation
bits16 = torch.arange(1, 0x7C00, dtype=torch.int32).to(torch.uint16)
vals = torch.tensor(bits16.view(torch.uint16).numpy(), dtype=torch.float32)  # placeholder replaced below
# simpler safe path: enumerate positive fp16 values by viewing uint16 bits
u16 = torch.arange(1, 0x7C00, dtype=torch.int16).to(torch.uint16) if False else None
v = torch.arange(1, 0x7C00, dtype=torch.int16)  # keep; do the real construction next
# real construction: use torch.frombuffer-free approach via int16->uint16 bitcast
raw = torch.arange(1, 0x7C00, dtype=torch.int32)
raw16 = (torch.arange(1, 0x7C00, dtype=torch.uint16))  # may not exist on older torch; fallback chain
try:
    fp = raw16.view(torch.float16).to(torch.float64)
except Exception:
    fp = torch.arange(1, 0x7C00, dtype=torch.int16).numpy().astype('uint16').view('float16').astype('float64')
    fp = torch.from_numpy(fp)
mask = (fp > 0) & (fp <= 0.03125)
scales = fp[mask]
S = scales.to(torch.float32)[None, None, :]
kern = ((qz * S).to(torch.float16)).view(torch.uint16).to(torch.int64) if hasattr(torch, 'uint16') else None
if kern is None:
    k16 = (qz * S).to(torch.float16).view(torch.int16).to(torch.int64)
    r16 = (qz.to(torch.float64) * scales.to(torch.float64)[None, None, :]).to(torch.float16).view(torch.int16).to(torch.int64)
    mism = (k16 != r16)
else:
    r16 = (qz.to(torch.float64) * scales.to(torch.float64)[None, None, :]).to(torch.float16).view(torch.uint16).to(torch.int64)
    mism = (kern != r16)
sub_n = int((scales.view(torch.int16).to(torch.int64) < 0x0400).sum().item()) if not hasattr(torch, 'uint16') else int((scales.view(torch.uint16) < 0x0400).sum().item())
print(json.dumps({
  "claim": "c3 decoded-weight H16 rounding",
  "n_scales_tested": int(mask.sum().item()),
  "n_subnormal_scales": sub_n,
  "n_combinations": int(16*16*int(mask.sum().item())),
  "n_mismatches": int(mism.sum().item()),
  "bit_identical": bool(not bool(mism.any().item()))}))