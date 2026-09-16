
import torch, json, importlib.util, traceback
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_23/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res={}
for C in [3, 48, 96, 64]:
    try:
        x = torch.randn(2,C,4,4, device='cuda').to(memory_format=torch.channels_last)
        s = torch.randn(C, device='cuda')
        out = m.scale_channels(x, s, C)
        ref = x*s.view(1,C,1,1)
        res[str(C)] = dict(ok=True, max_abs_err=(out-ref).abs().max().item())
    except Exception as e:
        res[str(C)] = dict(ok=False, err=type(e).__name__, msg=str(e)[:400])
print(json.dumps(res, indent=1))
