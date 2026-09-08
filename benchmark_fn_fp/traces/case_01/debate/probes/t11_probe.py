
import json, importlib.util, time, traceback, torch, triton
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_01/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(2)
dev="cuda"
out={"gpu": torch.cuda.get_device_name(0), "triton": triton.__version__}
def unclamped_ref(t,d,iq):
    s=(t-d)*iq
    mx=s.max(dim=1,keepdim=True).values
    return (s==mx).float().argmax(dim=1)
for V in [1024, 32000, 128256]:
    key=f"V{V}"
    B=4
    try:
        t=torch.softmax(torch.randn(B,V,device=dev),dim=1)
        d=torch.softmax(torch.randn(B,V,device=dev),dim=1)
        iq=1.0/torch.empty(B,V,device=dev).exponential_(1.0)
        t0=time.time()
        k=m.sample_recovered_tokens(t,d,iq)
        torch.cuda.synchronize()
        dt=time.time()-t0
        out[key+"_ok"]=True
        out[key+"_block_size"]=triton.next_power_of_2(V)
        out[key+"_secs"]=round(dt,3)
        out[key+"_match_unclamped"]=int((k==unclamped_ref(t,d,iq)).sum())
        out[key+"_rows"]=B
        out[key+"_in_range"]=bool(((k>=0)&(k<V)).all())
    except Exception as e:
        out[key+"_ok"]=False
        out[key+"_block_size"]=triton.next_power_of_2(V)
        out[key+"_err"]=repr(e)[:500]
        out[key+"_tb"]=traceback.format_exc()[-600:]
print(json.dumps(out))
