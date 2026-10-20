import torch, sys, json
sys.path.insert(0, "/root/cases/case_165")
import kernel

all_bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
finite = all_bits[torch.isfinite(all_bits) & (all_bits > torch.tensor(-1.0, dtype=torch.float16))]
n = finite.numel()
x = finite.to("cuda")
out = kernel.run(x)
ref = torch.log1p(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
max_err = err.max().item()
i = int(err.argmax())
res = {"metric":"max_abs_error_exhaustive_all_legal_fp16","legal_values":n,
       "max_abs_err":max_err,"violations":int((err>1e-6).sum()),
       "worst_x":finite[i].item(),"ref":ref[i].item(),"actual":out[i].item()}
print(json.dumps(res))
