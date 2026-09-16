
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_23/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

res = {}
torch.manual_seed(0)
for (N,C,H,W) in [(2,64,8,8),(1,32,4,4),(3,128,5,7)]:
    x_nchw = torch.randn(N,C,H,W, device='cuda')
    x = x_nchw.to(memory_format=torch.channels_last)
    scale = torch.randn(C, device='cuda')
    out = m.scale_channels(x, scale, C)
    ref = x * scale.view(1,C,1,1)
    err = (out.float()-ref.float()).abs().max().item()
    res[f"NHWC_{N}x{C}x{H}x{W}"] = dict(max_abs_err=err,
        out_is_cl=out.is_contiguous(memory_format=torch.channels_last),
        out_contig=out.is_contiguous())
    # also plain contiguous NCHW input (kernel convention NOT honored) - informational
    out2 = m.scale_channels(x_nchw.contiguous(), scale, C)
    ref2 = x_nchw * scale.view(1,C,1,1)
    res[f"NCHW_{N}x{C}x{H}x{W}_informational"] = (out2-ref2).abs().max().item()
print(json.dumps(res, indent=1))
