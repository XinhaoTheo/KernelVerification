
import json, importlib.util, torch
out={}
try:
    out["cuda"]=torch.cuda.is_available()
    spec=importlib.util.spec_from_file_location("k","/root/cases/case_12/kernel.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    torch.manual_seed(1)
    X=torch.randn(8,1024,device="cuda").float()
    res={}
    for eps in [1e-6,1e-5,1e-3,1e-2,1e-1]:
        Y=m.rms_norm_forward(X,eps)
        Xd=X.double()
        ms=(Xd*Xd).sum(dim=1,keepdim=True)/Xd.shape[1]
        ref=Xd/torch.sqrt(ms+eps)
        Yd=Y.double()
        rel=((Yd-ref).abs()/ref.abs().clamp_min(1e-300))
        res[str(eps)]={
          "max_rel_err":float(rel.max()),
          "mean_rel_err":float(rel.mean()),
          "max_abs_err":float((Yd-ref).abs().max()),
          "allclose_rtol1e-3_atol1e-5":bool(torch.allclose(Yd,ref,rtol=1e-3,atol=1e-5)),
          "allclose_rtol1e-2_atol1e-2":bool(torch.allclose(Yd,ref,rtol=1e-2,atol=1e-2)),
          "row0_rms":float(torch.sqrt(ms)[0,0]),
        }
    out["by_eps"]=res
    out["ok"]=True
except Exception as e:
    out["ok"]=False; out["error"]=repr(e)
print(json.dumps(out))
