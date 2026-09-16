
import json, importlib.util, torch, traceback
spec=importlib.util.spec_from_file_location("k","/root/cases/case_22/kernel.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
out={"dists":[], "large_K":[]}
torch.manual_seed(3)
K=1000003
# 1. catastrophic cancellation: near-orthogonal, sum ~ 0 with large terms
a=torch.randn(K,device="cuda")*1e3
b=torch.randn(K,device="cuda")*1e3
for name,(x,y) in {
  "large_scale_randn":(a,b),
  "cancel_antisym":(torch.cat([a[:K//2], -a[:K//2], a[K-1:]]), torch.cat([b[:K//2], b[:K//2], b[K-1:]])),
  "mixed_magnitude":(torch.where(torch.arange(K,device="cuda")%2==0, torch.full((K,),1e6,device="cuda"), torch.full((K,),1e-6,device="cuda")), torch.ones(K,device="cuda")),
  "all_positive":(torch.rand(K,device="cuda")+1.0, torch.rand(K,device="cuda")+1.0),
}.items():
    x=x.contiguous().float(); y=y.contiguous().float()
    n=x.numel()
    v=m.splitk_dot(x,y[:n]).item()
    ref=torch.dot(x.double(),y[:n].double()).item()
    sumabs=torch.sum((x.double()*y[:n].double()).abs()).item()
    out["dists"].append({"dist":name,"K":n,"kernel":v,"fp64_ref":ref,"abs_err":abs(v-ref),
        "rel_err_vs_ref":abs(v-ref)/max(abs(ref),1e-30),
        "rel_err_vs_sumabs":abs(v-ref)/max(sumabs,1e-30),"sum_abs_terms":sumabs})
# 2. large K: index arithmetic start+chunk in int32, and chunk >> BLOCK
for K in [1<<24, 100000003, 160000001]:
    try:
        o=torch.ones(K,device="cuda",dtype=torch.float32)
        cnt=m.splitk_dot(o,o).item()   # exact only up to 2^24; use fp64 compare below
        ref_cnt=float(K)
        # scaled ones so fp32 accumulation stays exact-ish: use 1/1024 values? instead compare relative
        out["large_K"].append({"K":K,"chunk":-(-K//512),"ones_result":cnt,"expected":ref_cnt,
                               "rel_deficit":(cnt-ref_cnt)/ref_cnt})
        del o; torch.cuda.empty_cache()
    except Exception as e:
        out["large_K"].append({"K":K,"error":repr(e)})
out["worst_rel_err_vs_sumabs"]=max(d["rel_err_vs_sumabs"] for d in out["dists"])
out["worst_large_K_rel_deficit"]=max((abs(r.get("rel_deficit",0.0)) for r in out["large_K"]), default=None)
print(json.dumps(out))
