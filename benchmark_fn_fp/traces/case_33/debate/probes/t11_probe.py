
import torch, numpy as np, importlib.util, json, traceback
torch.backends.cuda.matmul.allow_tf32=False
spec=importlib.util.spec_from_file_location("k","/root/cases/case_33/kernel.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def pack_k(q):
    K,N=q.shape; Kp=K//8
    out=np.zeros((Kp,N),np.uint32); qr=q.reshape(Kp,8,N).astype(np.uint32)
    for j in range(8): out |= (qr[:,j,:]<<(4*j))
    return out.view(np.int32)
def pack_n(z):
    G,N=z.shape; Nc=N//8
    out=np.zeros((G,Nc),np.uint32); zr=z.reshape(G,Nc,8).astype(np.uint32)
    for j in range(8): out |= (zr[:,:,j]<<(4*j))
    return out.view(np.int32)

res={}
def case(M,N,K,gs,seed):
    rng=np.random.default_rng(seed)
    q=rng.integers(0,16,size=(K,N)).astype(np.int64)
    G=(K+gs-1)//gs
    z=rng.integers(0,15,size=(G,N)).astype(np.int64)
    s=(rng.random((G,N))*0.2+0.02).astype(np.float32)
    torch.manual_seed(seed)
    a=torch.randn(M,K,device='cuda',dtype=torch.float32)
    bp=torch.from_numpy(pack_k(q)).cuda().contiguous()
    zp=torch.from_numpy(pack_n(z)).cuda().contiguous()
    st=torch.from_numpy(s).cuda().contiguous()
    qt=torch.from_numpy(q).cuda().float(); zt=torch.from_numpy(z).cuda().float()
    gc=torch.arange(K,device='cuda')//gs
    ng=K//gs
    gk=torch.clamp(torch.arange(K,device='cuda')//gs,max=ng-1)
    mapping_identical=bool(torch.equal(gc,gk))
    deq=(qt-(zt[gc]+1))*st[gc]
    ref=a@deq
    o1=m.gptq_matmul(a,bp,st,zp,gs,4); torch.cuda.synchronize()
    o2=m.gptq_matmul(a,bp,st,zp,gs,4); torch.cuda.synchronize()
    d=(o1-ref).abs()
    return dict(M=M,N=N,K=K,gs=gs,K_mod_BLOCKK=K%16,n_k_tiles=-(-K//16),
        group_mapping_matches_contract=mapping_identical,
        ref_absmax=float(ref.abs().max()),
        max_abs_err=float(d.max()), rel_err=float(d.max()/ref.abs().max()),
        mean_abs_err=float(d.mean()),
        allclose=bool(torch.allclose(o1,ref,rtol=1e-3,atol=1e-3)),
        deterministic_across_runs=bool(torch.equal(o1,o2)),
        out_has_nan=bool(torch.isnan(o1).any()), out_absmax=float(o1.abs().max()))
try:
    res['tail_K40_gs8']=case(64,64,40,8,2)
    res['tail_K40_gs8_seed3']=case(64,64,40,8,3)
    res['control_K32_gs8']=case(64,64,32,8,2)
    res['status']="ran"
except Exception as e:
    res['status']="exception"; res['err']=repr(e); res['tb']=traceback.format_exc()[-1500:]
print(json.dumps(res))
