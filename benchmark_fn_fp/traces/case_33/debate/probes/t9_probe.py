
import torch, numpy as np, importlib.util, json
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

def run_case(M,N,K,gs,seed=0):
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
    ar=torch.arange(K,device='cuda')
    gc=ar//gs
    ng=K//gs
    gk=torch.clamp(ar//gs,max=ng-1)
    deq_c=(qt-(zt[gc]+1))*st[gc]
    deq_k=(qt-(zt[gk]+1))*st[gk]
    ref_c=a@deq_c; ref_k=a@deq_k
    out=m.gptq_matmul(a,bp,st,zp,gs,4); torch.cuda.synchronize()
    d_c=(out-ref_c).abs(); d_k=(out-ref_k).abs()
    return dict(M=M,N=N,K=K,gs=gs,G_supplied=G,num_groups_kernel=ng,
                gidx_last8=gk[-8:].tolist(), gidx_contract_last8=gc[-8:].tolist(),
                ref_absmax=float(ref_c.abs().max()),
                max_abs_err_vs_contract=float(d_c.max()),
                rel_err_vs_contract=float(d_c.max()/ref_c.abs().max()),
                max_abs_err_vs_clamped_model=float(d_k.max()),
                rel_err_vs_clamped_model=float(d_k.max()/ref_c.abs().max()),
                allclose_contract=bool(torch.allclose(out,ref_c,rtol=1e-3,atol=1e-3)),
                allclose_clamped=bool(torch.allclose(out,ref_k,rtol=1e-3,atol=1e-3)))

res={}
res['bug_case_K48_gs32']=run_case(64,64,48,32)
res['control_K64_gs32']=run_case(64,64,64,32)
res['control_K32_gs32']=run_case(64,64,32,32)
print(json.dumps(res))
