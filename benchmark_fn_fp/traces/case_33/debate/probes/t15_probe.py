
import importlib.util, torch, numpy as np, json, math
spec = importlib.util.spec_from_file_location("kmod","/root/cases/case_33/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
dev='cuda'; torch.manual_seed(0)
M,N,K,gs=32,32,16,32
A=torch.randn(M,K)
q=torch.randint(0,16,(K,N),dtype=torch.int32)
ng=math.ceil(K/gs)   # 1
scales_row=torch.rand(ng,N)*0.1+0.01
zv=torch.randint(0,15,(ng,N),dtype=torch.int32)
qn=q.numpy().astype(np.uint32); packed=np.zeros((K//8,N),dtype=np.uint32)
for kk in range(K): packed[kk//8] |= (qn[kk]<<((kk%8)*4))
zvn=zv.numpy().astype(np.uint32); zp=np.zeros((ng,N//8),dtype=np.uint32)
for nn in range(N): zp[:,nn//8] |= (zvn[:,nn]<<((nn%8)*4))
b=torch.from_numpy(packed.astype(np.int32)).to(dev)
# Place a SENTINEL row immediately BEFORE the real scales/zeros row 0.
big_s=torch.empty(ng+1,N,device=dev); big_s[0]=7.0; big_s[1:]=scales_row.to(dev)
scales=big_s[1:]                       # row -1 in memory == 7.0 sentinel
big_z=torch.zeros(ng+1,N//8,dtype=torch.int32,device=dev)
big_z[0]=0                             # sentinel packed zeros -> nibble 0 -> zero=1
big_z[1:]=torch.from_numpy(zp.astype(np.int32)).to(dev)
zeros=big_z[1:]
res=dict(K=K,group_size=gs,sentinel_scale=7.0,sentinel_zero_nibble=0)
try:
    c=mod.gptq_matmul(A.to(dev),b,scales,zeros,gs); torch.cuda.synchronize(); c=c.cpu().double()
    r_correct=A.double()@((q.double()-(zv[0].double()+1.0))*scales_row[0].double())
    r_sentinel=A.double()@((q.double()-1.0)*7.0)
    res.update(crashed=False, out_absmax=float(c.abs().max()),
      max_abs_err_vs_correct_row0=float((c-r_correct).abs().max()),
      correct_absmax=float(r_correct.abs().max()),
      max_abs_err_vs_sentinel_rowminus1=float((c-r_sentinel).abs().max()),
      sentinel_absmax=float(r_sentinel.abs().max()),
      rel_err_vs_sentinel=float((c-r_sentinel).abs().max()/max(r_sentinel.abs().max(),1e-30)))
except Exception as e:
    res.update(crashed=True, err=repr(e)[:300])
print(json.dumps(res))
