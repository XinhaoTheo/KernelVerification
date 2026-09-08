
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_33/kernel.py")
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)

torch.manual_seed(3)
dev='cuda'
res={}
for (M,N,K,gs) in [(32,32,64,32),(64,64,128,32),(32,32,64,16)]:
    G = -(-K//gs)
    a = torch.randn(M,K,device=dev,dtype=torch.float32)
    b_packed = torch.randint(-2**31,2**31-1,(K//8,N),device=dev,dtype=torch.int32)
    zeros_packed = torch.randint(-2**31,2**31-1,(G,N//8),device=dev,dtype=torch.int32)
    scales = (torch.rand(G,N,device=dev)*0.1+0.01).contiguous()
    k_idx=torch.arange(K,device=dev); n_idx=torch.arange(N,device=dev)
    q=(b_packed[(k_idx//8),:] >> ((k_idx%8)*4).unsqueeze(1)) & 15
    zq=(zeros_packed[:,(n_idx//8)] >> ((n_idx%8)*4).unsqueeze(0)) & 15
    z=(zq+1).float()
    gmap=k_idx//gs
    c_ref = a @ ((q.float()-z[gmap])*scales[gmap])
    c = kmod.gptq_matmul(a,b_packed,scales,zeros_packed,gs,bits=4)
    torch.cuda.synchronize()
    err=(c-c_ref).abs().max().item(); den=c_ref.abs().max().item()
    res[f"M{M}_N{N}_K{K}_gs{gs}"]={"groups":G,"max_abs_err":err,"max_abs_ref":den,
        "max_rel_err":err/max(den,1e-12),"pass_1e-3":bool(err<1e-3)}
print(json.dumps({"metric":"max abs err vs exact grouped-dequant reference on aligned configs (K%gs==0, K%16==0)",
 "cases":res,"all_pass":all(v["pass_1e-3"] for v in res.values())}))
