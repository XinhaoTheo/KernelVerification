
import json, torch, importlib.util, traceback
res={}
try:
    spec=importlib.util.spec_from_file_location("k","/root/cases/case_04/kernel.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    torch.manual_seed(0)
    eps=1e-6
    n_rows,n_cols=8,512
    base=torch.randn(n_rows,n_cols,device="cuda")
    scales=torch.ones(n_rows,1,device="cuda")
    scales[4:]=1e-8            # near-zero-magnitude rows
    X=(base*scales).to(torch.bfloat16)
    Y=m.rms_norm_forward(X,eps)
    x32=X.float()
    ms=(x32*x32).sum(-1,keepdim=True)/n_cols
    rstd=torch.rsqrt(ms+eps)
    ref_bf16=(x32*rstd).to(torch.bfloat16)
    ref_f32=x32*rstd
    Yf=Y.float(); Rb=ref_bf16.float()
    def stats(a,b):
        d=(a-b).abs()
        rel=d/b.abs().clamp_min(1e-30)
        return {"max_abs":d.max().item(),"max_rel":rel.max().item(),
                "frac_rel_gt_1e-5":(rel>1e-5).float().mean().item()}
    norm_sl=slice(0,4); near_sl=slice(4,8)
    res={"ok":True,
      "kernel_out_dtype":str(Y.dtype),
      "vs_bf16_ref_all":stats(Yf,Rb),
      "vs_bf16_ref_normal_rows":stats(Yf[norm_sl],Rb[norm_sl]),
      "vs_bf16_ref_nearzero_rows":stats(Yf[near_sl],Rb[near_sl]),
      "vs_fp32_ref_all":stats(Yf,ref_f32),
      "kernel_rounded_to_bf16_exact_match_frac":(Y.to(torch.bfloat16)==ref_bf16).float().mean().item(),
      "allclose_vs_bf16ref_rtol1e-5":torch.allclose(Yf,Rb,rtol=1e-5,atol=1e-8),
      "allclose_vs_bf16ref_rtol1e-2":torch.allclose(Yf,Rb,rtol=1e-2,atol=1e-8),
      "allclose_vs_fp32ref_rtol1e-5":torch.allclose(Yf,ref_f32,rtol=1e-5,atol=1e-8),
      "rstd_nearzero_rows":rstd[near_sl].flatten().tolist(),
      "rstd_normal_rows":rstd[norm_sl].flatten().tolist(),
      "bf16_eps_2pow-8":2**-8}
except Exception as e:
    res={"ok":False,"err":traceback.format_exc()[-1500:]}
print(json.dumps(res))
