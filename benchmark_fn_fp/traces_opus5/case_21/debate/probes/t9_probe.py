
import json, importlib.util, traceback, torch
res={}
try:
    spec=importlib.util.spec_from_file_location("k","/root/cases/case_22/kernel.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    res["import_ok"]=True
    res["triton_version"]=__import__("triton").__version__
    torch.manual_seed(0)
    K=1<<20
    a=torch.randn(K,device="cuda",dtype=torch.float32)
    b=torch.randn(K,device="cuda",dtype=torch.float32)
    out=m.splitk_dot(a,b)
    ref=torch.dot(a.double(),b.double()).item()
    res["ran_without_exception"]=True
    res["out_shape"]=list(out.shape); res["out_dtype"]=str(out.dtype); res["out_device"]=str(out.device)
    v=out.item()
    res["K"]=K; res["kernel_value"]=v; res["fp64_ref"]=ref
    res["abs_err"]=abs(v-ref); res["rel_err"]=abs(v-ref)/max(abs(ref),1e-30)
    # magnitude scale for cancellation-aware tolerance
    res["sum_abs_terms"]=torch.sum((a.double()*b.double()).abs()).item()
    res["rel_err_vs_sumabs"]=abs(v-ref)/res["sum_abs_terms"]
    # repeat-run spread (allowed by contract, reported only)
    vals=[m.splitk_dot(a,b).item() for _ in range(5)]
    res["repeat_spread"]=max(vals)-min(vals)
except Exception as e:
    res["ran_without_exception"]=False
    res["error"]=repr(e); res["tb"]=traceback.format_exc()[-1500:]
print(json.dumps(res))
