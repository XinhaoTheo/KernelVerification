import torch, sys
sys.path.insert(0, '/root/cases/case_135')
import kernel

bits = torch.arange(65536, dtype=torch.int32)
fp16 = bits.to(torch.uint16).view(torch.float16)
finite = torch.isfinite(fp16)
codes = bits[finite]
xs = fp16[finite].to('cuda')  # ~65504*2? actually 65504 max, fits

# run in chunks to keep N<=131072 (fine, one chunk)
out = kernel.run(xs)
ref = xs.to(torch.float64) / torch.sqrt(1.0 + xs.to(torch.float64).double()**2)
err = (out.double() - ref).abs()
mx = err.max().item()
i = err.argmax().item()
worst_x = xs[i].item()
print({'n': xs.numel(), 'max_abs_err': mx, 'worst_x': worst_x,
       'n_over_1e-6': int((err > 1e-6).sum().item()),
       'top5': sorted(err.tolist(), reverse=True)[:5]})
