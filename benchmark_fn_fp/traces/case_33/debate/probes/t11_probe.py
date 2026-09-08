
import importlib.util, torch, numpy as np, json, math
spec = importlib.util.spec_from_file_location("kmod","/root/cases/case_33/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
dev='cuda'
torch.manual_seed(0)
M,N,K,gs=32,32,16,32
A=torch.randn(M,K)
q=torch.randint(0,16,(K,N),dtype=torch.int32)
ng=math.ceil(K/gs)
scales=torch.rand(ng,N)*0.1+0.01
zv=torch.randint(0,15,(ng,N),dtype=torch.int32)
qn=q.numpy().astype(np.uint32); packed=np.zeros((K//8,N),dtype=np.uint32)
for kk in range(K): packed[kk//8] |= (qn[kk]<<((kk%8)*4))
zvn=zv.numpy().astype(np.uint32); zp=np.zeros((ng,N//8),dtype=np.uint32)
for nn in range(N): zp[:,nn//8] |= (zvn[:,nn]<<((nn%8)*4))
b=torch.from_numpy(packed.astype(np.int32)); qz=torch.from_numpy(zp.astype(np.int32))
gi_wrapper=torch.clamp(torch.arange(K,dtype=torch.int32)//gs, max=K//gs-1)
res=dict(K=K,group_size=gs,rows_supplied=ng,wrapper_num_groups=K//gs,
         wrapper_g_idx_unique=sorted(set(gi_wrapper.tolist())))
z0=zv[0].double()+1.0; s0=scales[0].double()
r=A.double()@((q.double()-z0)*s0)
try:
    c=mod.gptq_matmul(A.to(dev),b.to(dev),scales.to(dev),qz.to(dev),gs)
    torch.cuda.synchronize(); c=c.cpu().double()
    c2=mod.gptq_matmul(A.to(dev),b.to(dev),scales.to(dev),qz.to(dev),gs)
    torch.cuda.synchronize(); c2=c2.cpu().double()
    res.update(crashed=False,
      max_abs_err_vs_contract=float((c-r).abs().max()),
      max_rel_err_vs_contract=float((c-r).abs().max()/r.abs().max()),
      ref_absmax=float(r.abs().max()), out_absmax=float(c.abs().max()),
      run_to_run_max_diff=float((c-c2).abs().max()),
      all_zero_output=bool(c.abs().max()==0))
except Exception as e:
    res.update(crashed=True, err=repr(e)[:300])
print(json.dumps(res))
