
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

out_json={}
try:
    M,N,K,gs=64,64,16,32   # K multiple of BLOCK_SIZE_K=16 -> no k-tail confound
    rng=np.random.default_rng(1)
    q=rng.integers(0,16,size=(K,N)).astype(np.int64)
    G=(K+gs-1)//gs   # == 1
    # real group row 0 + one SENTINEL row stored physically BEFORE it
    z_all=np.stack([np.full((N,),3,dtype=np.int64), rng.integers(0,15,size=(N,)).astype(np.int64)])
    s_all=np.stack([np.full((N,),5.0,dtype=np.float32), (rng.random((N,))*0.2+0.02).astype(np.float32)])
    zp_all=torch.from_numpy(pack_n(z_all)).cuda().contiguous()
    st_all=torch.from_numpy(s_all).cuda().contiguous()
    zp=zp_all[1:]; st=st_all[1:]         # supplied tensors: row0 == real group 0
    torch.manual_seed(1)
    a=torch.randn(M,K,device='cuda',dtype=torch.float32)
    bp=torch.from_numpy(pack_k(q)).cuda().contiguous()
    qt=torch.from_numpy(q).cuda().float()
    zt_all=torch.from_numpy(z_all).cuda().float(); s_allt=torch.from_numpy(s_all).cuda()
    # contract: every k uses supplied row 0 (= physical row 1)
    deq_c=(qt-(zt_all[1]+1))*s_allt[1]
    # sentinel model: every k uses physical row 0 (below base)
    deq_s=(qt-(zt_all[0]+1))*s_allt[0]
    ref_c=a@deq_c; ref_s=a@deq_s
    ng=K//gs
    gidx=torch.clamp(torch.arange(K)//gs,max=ng-1)
    o=m.gptq_matmul(a,bp,st,zp,gs,4); torch.cuda.synchronize()
    o2=m.gptq_matmul(a,bp,st,zp,gs,4); torch.cuda.synchronize()
    out_json=dict(status="ran",M=M,N=N,K=K,gs=gs,G_supplied=G,num_groups_kernel=ng,
      gidx_unique=sorted(set(gidx.tolist())),
      scales_stride0=st.stride(0), zeros_stride0=zp.stride(0),
      ref_contract_absmax=float(ref_c.abs().max()),
      max_abs_err_vs_contract=float((o-ref_c).abs().max()),
      max_abs_err_vs_sentinel_belowbase=float((o-ref_s).abs().max()),
      allclose_contract=bool(torch.allclose(o,ref_c,rtol=1e-3,atol=1e-3)),
      allclose_sentinel=bool(torch.allclose(o,ref_s,rtol=1e-3,atol=1e-3)),
      deterministic_across_runs=bool(torch.equal(o,o2)),
      out_absmax=float(o.abs().max()), out_has_nan=bool(torch.isnan(o).any()))
except Exception as e:
    out_json=dict(status="exception",err=repr(e),tb=traceback.format_exc()[-1200:])
print(json.dumps(out_json))
