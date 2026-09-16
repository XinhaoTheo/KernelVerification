
import json, torch, importlib.util, traceback
res={}
try:
    spec=importlib.util.spec_from_file_location("k","/root/cases/case_04/kernel.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    torch.manual_seed(0)
    eps=1e-6
    out={}
    for name,dt in [("fp32",torch.float32),("bf16",torch.bfloat16),("fp16",torch.float16)]:
        X=torch.randn(4,512,device="cuda").to(dt)
        Y=m.rms_norm_forward(X,eps)
        # reference: liger-style, fp32 accumulate, store in input dtype
        x32=X.float()
        ms=(x32*x32).sum(-1,keepdim=True)/x32.shape[-1]
        ref=(x32*torch.rsqrt(ms+eps)).to(dt)
        strict_ok=True; strict_msg=""
        try:
            torch.testing.assert_close(Y,ref)
        except Exception as e:
            strict_ok=False; strict_msg=str(e).splitlines()[0][:200]
        out[name]={"in_dtype":str(X.dtype),"out_dtype":str(Y.dtype),"ref_dtype":str(ref.dtype),
                   "dtype_matches_input":Y.dtype==X.dtype,
                   "assert_close_strict_ok":strict_ok,"assert_close_msg":strict_msg}
    res={"ok":True,"per_dtype":out}
except Exception as e:
    res={"ok":False,"err":traceback.format_exc()[-1500:]}
print(json.dumps(res))
