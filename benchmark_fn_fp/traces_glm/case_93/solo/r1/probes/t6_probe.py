import torch, sys
sys.path.insert(0, "/root/cases/case_165")
import kernel

# all finite FP16 values strictly greater than -1
all_bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
finite = all_bits[torch.isfinite(all_bits) & (all_bits > torch.tensor(-1.0, dtype=torch.float16))]
n = finite.numel()
print("legal values:", n)

# reference on exact stored values
ref = torch.log1p(finite.to(torch.float64))

# run kernel in chunks under 131072 (all fit at once)
x = finite.to("cuda")
out = kernel.run(x)
err = (out.to(torch.float64) - ref).abs()
max_err = err.max().item()
i = int(err.argmax())
print("max abs err:", max_err)
print("at value:", finite[i].item(), "ref:", ref[i].item(), "actual:", out[i].item())
print("count > 1e-6:", int((err > 1e-6).sum()))
import json
print(json.dumps({"metric":"max_abs_error_exhaustive","legal_values":n,"max_abs_err":max_err,"violations":int((err>1e-6).sum()),"worst_x":finite[i].item()}))
