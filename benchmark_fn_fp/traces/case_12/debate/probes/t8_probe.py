
import json, importlib.util, torch
out={}
try:
    out["cuda"]=torch.cuda.is_available()
    spec=importlib.util.spec_from_file_location("k","/root/cases/case_12/kernel.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    torch.manual_seed(0)
    res={}
    for scale in [1e-3, 1e-2, 1e-1, 1.0]:
        X=(scale*torch.randn(4,512,device="cuda")).float()
        eps=1e-6
        Y=m.rms_norm_forward(X,eps)
        Xd=X.double()
        ms=(Xd*Xd).sum(dim=1,keepdim=True)/Xd.shape[1]
        ref=Xd/torch.sqrt(ms+eps)
        Yd=Y.double()
        denom=ref.abs().clamp_min(1e-300)
        relerr=((Yd-ref).abs()/denom)
        res[str(scale)]={
          "row_rms":float(torch.sqrt(ms)[0,0]),
          "max_abs_err":float((Yd-ref).abs().max()),
          "max_rel_err":float(relerr.max()),
          "mean_rel_err":float(relerr.mean()),
          "allclose_rtol1e-3":bool(torch.allclose(Yd,ref,rtol=1e-3,atol=1e-5)),
        }
    out["by_scale"]=res
    # zero row check
    Xz=torch.zeros(2,512,device="cuda"); Xz[1,0]=1e-7
    Yz=m.rms_norm_forward(Xz,1e-6)
    out["zero_row_maxabs_Y"]=float(Yz.abs().max())
    out["tiny_row_expected_ref_max"]=float((1e-7/ (( (1e-7**2)/512 + 1e-6)**0.5)))
    out["ok"]=True
except Exception as e:
    out["ok"]=False; out["error"]=repr(e)
print(json.dumps(out))
