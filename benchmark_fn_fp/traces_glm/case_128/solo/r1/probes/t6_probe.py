import torch, math, json, sys
sys.path.insert(0, "/root/cases/case_128")
import kernel

# all finite FP16 values: every 16-bit pattern, then drop inf/nan
bits = torch.arange(0, 65536, dtype=torch.int32).to(torch.uint16)
x_all = bits.view(torch.float16).view(-1)
finite_mask = torch.isfinite(x_all)
x = x_all[finite_mask]
print("num finite fp16 values:", x.numel())

x_cuda = x.to("cuda")
out = kernel.run(x_cuda)
ref = kernel.reference(x_cuda).to(torch.float64)

diff = (out.to(torch.float64) - ref).abs()
max_err = diff.max().item()
worst_i = int(diff.argmax())
print("max abs err:", max_err, "at x =", x[worst_i].item(), "out:", out[worst_i].item(), "ref:", ref[worst_i].item())
viol = (diff > 1e-6)
print("num violations >1e-6:", int(viol.sum()))
if viol.any():
    idx = viol.nonzero().flatten()
    print("first few violating x:", x[idx[:10]].tolist())
    print("their errors:", [float(diff[i]) for i in idx[:10]])
# also test random input via make_inputs and edge sizes
for size in (1, 4093, 131072):
    xt = kernel.make_inputs(size=size)[0]
    o = kernel.run(xt); r = kernel.reference(xt)
    d = (o.to(torch.float64)-r).abs().max().item()
    print(f"make_inputs size={size}: max err {d}")
print(json.dumps({"max_abs_err_full_domain": max_err, "violations": int(viol.sum()), "worst_x": x[worst_i].item()}))
