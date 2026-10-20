import torch, numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_163/kernel.py")
kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)

dev = 'cuda'
raw = np.arange(0, 65536, dtype=np.uint16)
x_all = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)
inrange = (x_all.abs() <= 10) & torch.isfinite(x_all)
legal = x_all[inrange].to(dev)
print("num legal:", legal.numel())

mx_rel = 0.0; worst_x = None; worst_out = None; worst_ref = None; bad_count = 0
for s in range(0, legal.numel(), 131072):
    c = legal[s:s+131072].contiguous()
    out = kernel.run(c)
    ref = torch.exp2(c.to(torch.float64))
    rel = (out.to(torch.float64) - ref).abs() / ref
    bad_count += int((rel > 1e-6).sum().item())
    j = rel.argmax().item()
    if rel[j].item() > mx_rel:
        mx_rel = rel[j].item()
        worst_x = c[j].item(); worst_out = out[j].item(); worst_ref = ref[j].item()
print(json.dumps({"probe": "c2_kernel_full_domain",
  "num_legal_bf16": int(legal.numel()),
  "max_rel_err_kernel": float(mx_rel),
  "count_exceed_1e-6": int(bad_count),
  "worst_x": float(worst_x), "worst_out": float(worst_out), "worst_ref": float(worst_ref),
  "tolerance": 1e-6}))
