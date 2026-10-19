
import json, sys, traceback, importlib.util, torch
res={}
try:
    import triton; res["triton_version"]=triton.__version__
except Exception as e:
    res["triton_version"]="ERR "+repr(e)
spec=importlib.util.spec_from_file_location("k","/root/cases/case_11/kernel.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res["cuda"]=torch.cuda.is_available()
S,H,D=8,4,64
torch.manual_seed(0)
x=torch.randn(S,H,D,device="cuda",dtype=torch.float32)
ang=torch.randn(S,D//2,device="cuda")
cos=torch.cos(ang).contiguous(); sin=torch.sin(ang).contiguous()
for flag in [False,True]:
    try:
        out=m.apply_rotary(x,cos,sin,flag)
        torch.cuda.synchronize()
        res[f"ran_interleaved_{flag}"]=True
        res[f"shape_{flag}"]=list(out.shape)
        res[f"finite_{flag}"]=bool(torch.isfinite(out).all().item())
    except Exception as e:
        res[f"ran_interleaved_{flag}"]=False
        res[f"err_{flag}"]=repr(e)[:600]
print(json.dumps(res))
