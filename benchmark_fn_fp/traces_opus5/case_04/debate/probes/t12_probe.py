
import json, torch, importlib.util, traceback
res={}
try:
    spec=importlib.util.spec_from_file_location("k","/root/cases/case_04/kernel.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    torch.manual_seed(1)
    eps=1e-6
    cases=[]
    for shape in [(2,64),(4,512),(3,1024),(5,37)]:
        for scale in [1.0, 1e-2, 1e-4, 1e-8, 1e-12, 1e2]:
            for dt in [torch.bfloat16, torch.float16, torch.float32]:
                X=(torch.randn(*shape,device="cuda")*scale).to(dt)
                Y=m.rms_norm_forward(X,eps)
                x32=X.float()
                ms=(x32*x32).sum(-1,keepdim=True)/shape[-1]
                rstd=torch.rsqrt(ms+eps)
                ref_dt=(x32*rstd).to(dt)
                ref_f32=x32*rstd
                # is divergence purely missing store rounding?
                exact_after_round=bool(torch.equal(Y.to(dt),ref_dt))
                d32=(Y.float()-ref_f32).abs()
                rel32=(d32/ref_f32.abs().clamp_min(1e-30))
                ddt=(Y.to(dt).float()-ref_dt.float()).abs()
                cases.append({"shape":str(shape),"scale":scale,"dtype":str(dt),
                  "out_dtype":str(Y.dtype),
                  "exact_after_round_to_input_dtype":exact_after_round,
                  "max_abs_vs_fp32ref":d32.max().item(),
                  "max_rel_vs_fp32ref":rel32.max().item(),
                  "max_abs_after_round_vs_dtyperef":ddt.max().item(),
                  "any_nan":bool(torch.isnan(Y).any().item())})
    n=len(cases)
    all_exact=all(c["exact_after_round_to_input_dtype"] for c in cases)
    worst_rel32=max(c["max_rel_vs_fp32ref"] for c in cases)
    bad=[c for c in cases if not c["exact_after_round_to_input_dtype"]]
    fp32_cases=[c for c in cases if c["dtype"]=="torch.float32"]
    res={"ok":True,"n_cases":n,
      "all_exact_after_round_to_input_dtype":all_exact,
      "n_not_exact":len(bad),"not_exact_examples":bad[:6],
      "worst_max_rel_vs_fp32ref_all_cases":worst_rel32,
      "worst_max_rel_vs_fp32ref_fp32_inputs":max(c["max_rel_vs_fp32ref"] for c in fp32_cases),
      "any_nan_any_case":any(c["any_nan"] for c in cases),
      "all_out_dtype_fp32":all(c["out_dtype"]=="torch.float32" for c in cases)}
except Exception as e:
    res={"ok":False,"err":traceback.format_exc()[-1500:]}
print(json.dumps(res))
