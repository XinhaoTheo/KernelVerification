
import torch, kernel, json
# all finite fp16 values
bits = torch.arange(0, 65536, dtype=torch.int32)
x = torch.tensor(bits.numpy().view('float16') if False else bits, dtype=torch.int32)
x = torch.tensor(bits.view(torch.int16).numpy(), dtype=torch.int16).view(torch.float16)
finite = torch.isfinite(x)
x = x[finite]
print("count", x.numel())
out = kernel.run(x.cuda())
ref = kernel.reference(x.cuda())
err = (out.double() - ref).abs()
tol = 1e-6
bad = err > tol
print("max_err", err.max().item())
print("n_bad", bad.sum().item())
if bad.any():
    idx = torch.where(bad)[0]
    print("worst x:", x[idx].tolist()[:10])
    print("worst err:", err[idx].tolist()[:10])
print(json.dumps({"max_err": err.max().item(), "n_over_tol": int(bad.sum()), "n_tested": int(x.numel()), "tol": tol}))
