
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_07/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(1)
dev='cuda'
res={}
for (M,K,N,group,actorder) in [(70,256,64,64,False),(128,256,128,128,True),(32,128,32,32,True)]:
    G=K//group
    q=torch.randint(0,16,(K,N),device=dev,dtype=torch.int32)
    zs=torch.randint(0,16,(G,N),device=dev,dtype=torch.int32)
    scales=(torch.rand((G,N),device=dev,dtype=torch.float16)*0.1+0.01)
    if actorder:
        base=(torch.arange(K,device=dev)//group).to(torch.int32)
        perm=torch.randperm(K,device=dev)
        g_idx=base[perm].contiguous()
    else:
        g_idx=(torch.arange(K,device=dev)//group).to(torch.int32)
    qw=torch.zeros((K//8,N),device=dev,dtype=torch.int32)
    qr=q.reshape(K//8,8,N)
    for j in range(8): qw |= (qr[:,j,:]&0xF)<<(4*j)
    qz=torch.zeros((G,N//8),device=dev,dtype=torch.int32)
    zr=zs.reshape(G,N//8,8)
    for j in range(8): qz |= (zr[:,:,j]&0xF)<<(4*j)

    a=(torch.randn((M,K),device=dev,dtype=torch.float16))
    deq=(q.float()-(zs+1)[g_idx.long()].float())*scales[g_idx.long()].float()
    ref=a.float()@deq
    c=m.gptq_matmul(a,qw,scales,qz,g_idx,bits=4).float()
    err=(c-ref).abs()
    den=ref.abs().clamp_min(1e-3)
    res[f"M{M}_K{K}_N{N}_g{group}_ao{int(actorder)}"]={
      "max_abs":float(err.max()),
      "max_rel":float((err/den).max()),
      "ref_absmax":float(ref.abs().max()),
      "rel_fro":float(err.norm()/ref.norm()),
      "rows_bad_gt_5pct":[int(i) for i in torch.nonzero((err/den).max(1).values>0.05).flatten()[:10]],
    }
print(json.dumps(res))
