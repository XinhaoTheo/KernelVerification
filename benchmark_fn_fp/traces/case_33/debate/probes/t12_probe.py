
import importlib.util, json
import numpy as np, torch

spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_33/kernel.py")
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)

def build(M,K,N,gs,seed=0):
    g = torch.Generator().manual_seed(seed)
    ng = (K+gs-1)//gs
    q = torch.randint(0,16,(K,N),generator=g,dtype=torch.int64)
    z = torch.randint(1,17,(ng,N),generator=g,dtype=torch.int64)
    s = torch.rand((ng,N),generator=g)*0.1+0.01
    a = torch.randn((M,K),generator=g)
    qn = q.numpy().astype(np.uint32)
    bp = np.zeros((K//8,N),dtype=np.uint32)
    for k in range(K):
        bp[k//8] |= qn[k] << (4*(k%8))
    b = torch.from_numpy(bp.view(np.int32).copy()).cuda()
    zn = (z.numpy()-1).astype(np.uint32)
    qz = np.zeros((ng,N//8),dtype=np.uint32)
    for n in range(N):
        qz[:,n//8] |= zn[:,n] << (4*(n%8))
    qzp = torch.from_numpy(qz.view(np.int32).copy()).cuda()
    return dict(a=a.cuda(), b=b, s=s.cuda(), qzp=qzp, q=q, z=z, sc=s, ac=a, ng=ng)

def ref_with(d,K,gidx):
    deq = (d["q"].double() - d["z"][gidx].double()) * d["sc"][gidx].double()
    return d["ac"].double() @ deq

res = {}
for tag,(M,K,N,gs) in {"target":(64,96,64,64), "control":(64,128,64,32)}.items():
    d = build(M,K,N,gs,seed=7)
    ng_true = (K+gs-1)//gs
    gidx_true = torch.arange(K)//gs
    ng_floor = K//gs
    gidx_clamp = torch.clamp(torch.arange(K)//gs, max=ng_floor-1)
    R_true = ref_with(d,K,gidx_true)
    R_clamp = ref_with(d,K,gidx_clamp)
    out = kmod.gptq_matmul(d["a"], d["b"], d["s"], d["qzp"], gs, 4).double().cpu()
    res[tag] = {
      "config": {"M":M,"K":K,"N":N,"group_size":gs,"ceil_groups":ng_true,"floor_groups":ng_floor,
                 "K_mult_of_16":K%16==0,"K_mult_of_gs":K%gs==0},
      "ref_max_abs": R_true.abs().max().item(),
      "max_abs_err_vs_contract_ref": (out-R_true).abs().max().item(),
      "rel_err_vs_contract_ref": ((out-R_true).abs().max()/R_true.abs().max()).item(),
      "max_abs_err_vs_clamped_ref": (out-R_clamp).abs().max().item(),
      "n_gidx_mismatched_cols": int((gidx_true!=gidx_clamp).sum().item()),
    }
res["metric_reason"] = "continuous matmul output; max-abs error vs contract reference (ceil groups) and vs clamped-group reference identifies which mapping the kernel implements"
print(json.dumps(res))
