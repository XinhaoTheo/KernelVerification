
import importlib.util, torch, numpy as np, json, math, traceback
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
    return d['A'].double()@((d['q'].double()-z)*s)

out={}
cases={'tail_only_K72_gs24':(32,32,72,24),'claim_K72_gs32':(32,32,72,32)}
for name,(M,N,K,gs) in cases.items():
    try:
        d=build(M,N,K,gs)
        A=d['A'].to(dev); B=d['b'].to(dev); S=d['scales'].to(dev); Z=d['qz'].to(dev)
        c1=mod.gptq_matmul(A,B,S,Z,gs); torch.cuda.synchronize(); c1=c1.cpu().double()
        # scribble GPU memory to change what lies past the ends of A/B/g_idx
        junk=torch.full((1<<22,), 12345.0, device=dev); del junk
        junk2=torch.randn(1<<22, device=dev)*1e6; torch.cuda.synchronize(); del junk2
        c2=mod.gptq_matmul(A,B,S,Z,gs); torch.cuda.synchronize(); c2=c2.cpu().double()
        r_ceil=ref(d,K,gs,'ceil'); r_floor=ref(d,K,gs,'floor')
        out[name]=dict(K=K,group_size=gs,K_mod_BLOCK_K=K%16,rows_supplied=d['ng'],
          num_k_tiles=math.ceil(K/16),
          max_abs_err_vs_contract=float((c1-r_ceil).abs().max()),
          max_abs_err_vs_floorclamp_model=float((c1-r_floor).abs().max()),
          run_to_run_max_diff=float((c1-c2).abs().max()),
          ref_absmax=float(r_ceil.abs().max()),
          crashed=False)
    except Exception as e:
        out[name]=dict(crashed=True, err=repr(e)[:300])
        break
print(json.dumps(out))
