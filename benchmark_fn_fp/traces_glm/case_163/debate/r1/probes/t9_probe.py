import torch, numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_163/kernel.py")
kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)

dev = 'cuda'
raw = np.arange(0, 65536, dtype=np.uint16)
x_all = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)
inrange = (x_all.abs() <= 10) & torch.isfinite(x_all)
legal = x_all[inrange].to(dev)
print("num legal:", legal.numel())

chunks = []
for s in range(0, legal.numel(), 131072):
    c = legal[s:s+131072].contiguous()
    out = kernel.run(c)
    ref = torch.exp2(c.to(torch.float64))
    rel = (out.to(torch.float64) - ref).abs() / ref
    bad = rel > 1e-6
    j = rel.argmax()
    chunks.append((rel[j].item(), c[j].item(), out[j].item(), ref[j].item(), int(bad.sum())))
mx = max(ch[0] for c in chunks)
print(json.dumps({"probe": "c2_kernel_full_domain", "num_legal_bf16": int(legal.numel()),
                  "max_rel_err_kernel": float(mx),
                  "count_exceed_1e-6": int(sum(c[4] for c in chunks)),
                  "worst_chunks": chunks[:2]}))
