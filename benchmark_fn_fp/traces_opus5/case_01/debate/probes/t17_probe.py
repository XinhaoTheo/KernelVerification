
import json, traceback, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_01/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import triton
dev='cuda'; torch.manual_seed(3)
res={"gpu": torch.cuda.get_device_name(0)}
props = torch.cuda.get_device_properties(0)
res["shared_mem_per_block"] = getattr(props, "shared_memory_per_block", None)

def trial(V, B=4):
    d={"vocab_size":V, "BLOCK_SIZE":triton.next_power_of_2(V)}
    try:
        target = torch.softmax(torch.randn(B,V,device=dev), dim=-1)
        draft  = torch.softmax(torch.randn(B,V,device=dev), dim=-1)
        q = torch.empty(B,V,device=dev).exponential_(1.0); inv_q = 1.0/q
        out = m.sample_recovered_tokens(target, draft, inv_q)
        torch.cuda.synchronize()
        d["raised"]=False
        d["out"]=out.tolist()
        d["dtype"]=str(out.dtype)
        d["in_range"]=bool(((out>=0)&(out<V)).all())
        sc = torch.clamp(target-draft, min=0.0)*inv_q
        ref = sc.argmax(dim=1)
        d["ref"]=ref.tolist()
        d["index_match"]=bool((out==ref).all())
        ar=torch.arange(B,device=dev)
        d["ref_score_at_ref"]=[float(x) for x in sc[ar,ref]]
        d["ref_score_at_kernel"]=[float(x) for x in sc[ar,out]]
        d["max_score_gap"]=float((sc[ar,ref]-sc[ar,out]).max())
        resid = target-draft
        d["kernel_selected_residual_min"]=float(resid[ar,out].min())
        d["kernel_all_positive_residual"]=bool((resid[ar,out]>0).all())
    except Exception as e:
        d["raised"]=True
        d["error_type"]=type(e).__name__
        d["error_msg"]=str(e)[:600]
    return d

for V in [4096, 32000, 128256]:
    res["V_%d"%V]=trial(V)
print(json.dumps(res))
