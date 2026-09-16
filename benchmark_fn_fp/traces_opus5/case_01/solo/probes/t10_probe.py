
import torch, json, importlib.util, traceback
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_01/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(2)
dev='cuda'
res={"gpu":torch.cuda.get_device_name(0),"cases":[]}
for V in [4096, 8192, 32000, 32768, 50257, 128256]:
    B=8
    entry={"V":V,"block":1<<(V-1).bit_length()}
    try:
        t=torch.softmax(torch.randn(B,V,device=dev),dim=1)
        d=torch.softmax(torch.randn(B,V,device=dev),dim=1)
        inv_q=1.0/torch.empty(B,V,device=dev).exponential_(1.0)
        o=k.sample_recovered_tokens(t,d,inv_q)
        torch.cuda.synchronize()
        r=torch.argmax(torch.clamp(t-d,min=0)*inv_q,dim=1)
        entry["ok"]=True
        entry["mismatch"]=int((o!=r).sum())
        entry["neg_resid_returned"]=int(((t-d).gather(1,o.view(-1,1)).squeeze(1)<0).sum())
    except Exception as e:
        entry["ok"]=False
        entry["error_type"]=type(e).__name__
        entry["error"]=str(e)[:600]
    res["cases"].append(entry)
print(json.dumps(res))
