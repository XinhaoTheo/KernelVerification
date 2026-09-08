
import importlib.util, torch, numpy as np, json, math
spec = importlib.util.spec_from_file_location("kmod","/root/cases/case_33/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
dev='cuda'
torch.manual_seed(0)

def build(M,N,K,gs):
    A=torch.randn(M,K)
    q=torch.randint(0,16,(K,N),dtype=torch.int32)
    ng=math.ceil(K/gs)
    scales=torch.rand(ng,N)*0.1+0.01
    zv=torch.randint(0,15,(ng,N),dtype=torch.int32)
    qn=q.numpy().astype(np.uint32); packed=np.zeros((K//8,N),dtype=np.uint32)
    for kk in range(K): packed[kk//8] |= (qn[kk]<<((kk%8)*4))
    zvn=zv.numpy().astype(np.uint32); zp=np.zeros((ng,N//8),dtype=np.uint32)
    for nn in range(N): zp[:,nn//8] |= (zvn[:,nn]<<((nn%8)*4))
    return dict(A=A,q=q,scales=scales,zv=zv,ng=ng,
        b=torch.from_numpy(packed.astype(np.int32)),
        qz=torch.from_numpy(zp.astype(np.int32)))

def ref(d,K,gs,mode):
    ar=torch.arange(K)
    gi = ar//gs if mode=='ceil' else torch.clamp(ar//gs,max=max(K//gs-1,0))
    z=d['zv'][gi].double()+1.0; s=d['scales'][gi].double()
    deq=(d['q'].double()-z)*s
    return d['A'].double()@deq

out={}
for name,(M,N,K,gs) in {'control_K64_gs32':(32,32,64,32),'target_K80_gs32':(32,32,80,32)}.items():
    d=build(M,N,K,gs)
    c=mod.gptq_matmul(d['A'].to(dev),d['b'].to(dev),d['scales'].to(dev),d['qz'].to(dev),gs)
    torch.cuda.synchronize()
    c=c.cpu().double()
    r_ceil=ref(d,K,gs,'ceil'); r_floor=ref(d,K,gs,'floor')
    out[name]=dict(K=K,group_size=gs,rows_supplied=d['ng'],
      wrapper_num_groups=K//gs,
      max_abs_err_vs_contract=float((c-r_ceil).abs().max()),
      max_rel_err_vs_contract=float(((c-r_ceil).abs().max()/r_ceil.abs().max())),
      max_abs_err_vs_floorclamp_model=float((c-r_floor).abs().max()),
      ref_absmax=float(r_ceil.abs().max()),
      frac_elems_off_gt_1e3rel=float(((c-r_ceil).abs()>1e-3*r_ceil.abs().max()).double().mean()))
print(json.dumps(out))
