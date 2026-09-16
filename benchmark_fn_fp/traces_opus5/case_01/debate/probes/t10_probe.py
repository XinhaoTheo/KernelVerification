
import json, importlib.util, traceback, torch, triton
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_01/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(1)
dev="cuda"
out={"triton_version": triton.__version__, "torch_version": torch.__version__}

def unclamped_ref(t,d,iq):
    s=(t-d)*iq
    mx=s.max(dim=1,keepdim=True).values
    return (s==mx).float().argmax(dim=1)

# 1) wrapper path (single-iteration loop) with generic positive-residual inputs
B,V=16,256
t=torch.softmax(torch.randn(B,V,device=dev),dim=1)
d=torch.softmax(torch.randn(B,V,device=dev),dim=1)
iq=1.0/torch.empty(B,V,device=dev).exponential_(1.0)
try:
    k=m.sample_recovered_tokens(t,d,iq)
    out["wrapper_ok"]=True
    out["wrapper_matches_unclamped_argmax"]=int((k==unclamped_ref(t,d,iq)).sum())
    out["wrapper_rows"]=B
    out["wrapper_out_dtype"]=str(k.dtype)
    out["wrapper_in_range"]=bool(((k>=0)&(k<V)).all())
    out["wrapper_distinct_idx"]=len(set(k.tolist()))
except Exception as e:
    out["wrapper_ok"]=False
    out["wrapper_err"]=repr(e)[:400]
    out["wrapper_tb"]=traceback.format_exc()[-800:]

# 2) direct kernel call with BLOCK_SIZE < vocab_size -> multi-iteration loop-carried path
for BS in [64, 32]:
    key=f"direct_BS{BS}"
    try:
        o=torch.empty(B,dtype=torch.int64,device=dev)
        m.sample_recovered_tokens_kernel[(B,)](o,d,t,iq,V,BLOCK_SIZE=BS)
        torch.cuda.synchronize()
        ref=unclamped_ref(t,d,iq)
        out[key+"_ok"]=True
        out[key+"_match_count"]=int((o==ref).sum())
        out[key+"_rows"]=B
        out[key+"_kernel_idx"]=o.tolist()
        out[key+"_ref_idx"]=ref.tolist()
        out[key+"_all_in_first_block"]=bool((o<BS).all())
        out[key+"_wrapper_match"]=int((o==k).sum()) if out.get("wrapper_ok") else None
    except Exception as e:
        out[key+"_ok"]=False
        out[key+"_err"]=repr(e)[:400]
        out[key+"_tb"]=traceback.format_exc()[-800:]

# 3) non-power-of-2 vocab through wrapper (masked lanes present)
V2=300
t2=torch.softmax(torch.randn(B,V2,device=dev),dim=1)
d2=torch.softmax(torch.randn(B,V2,device=dev),dim=1)
iq2=1.0/torch.empty(B,V2,device=dev).exponential_(1.0)
try:
    k2=m.sample_recovered_tokens(t2,d2,iq2)
    out["npo2_ok"]=True
    out["npo2_match_unclamped"]=int((k2==unclamped_ref(t2,d2,iq2)).sum())
    out["npo2_rows"]=B
except Exception as e:
    out["npo2_ok"]=False; out["npo2_err"]=repr(e)[:400]
print(json.dumps(out))
