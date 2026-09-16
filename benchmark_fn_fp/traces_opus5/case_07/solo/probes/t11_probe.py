
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_07/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(7)
dev='cuda'
res={}
for (M,K,N,group) in [(256,1024,512,128),(64,512,256,32)]:
    G=K//group
    # realistic GPTQ quantization of continuous weights onto 16 levels per group
    W=torch.randn((K,N),device=dev,dtype=torch.float32)*0.02
    Wg=W.reshape(G,group,N)
    wmax=Wg.amax(dim=1); wmin=Wg.amin(dim=1)
    scale=((wmax-wmin)/15).clamp_min(1e-8).half()
    zero_full=(-wmin/scale.float()).round().clamp(0,15)      # true zero point (0..15)
    q=(Wg/scale.float().unsqueeze(1)+zero_full.unsqueeze(1)).round().clamp(0,15).to(torch.int32).reshape(K,N)
    stored=(zero_full-1).clamp(0,15).to(torch.int32)          # qzeros stores zero-1 (AutoGPTQ)
    zero_eff=stored+1
    g_idx=(torch.arange(K,device=dev)//group).to(torch.int32)
    qw=torch.zeros((K//8,N),device=dev,dtype=torch.int32); qr=q.reshape(K//8,8,N)
    for j in range(8): qw |= (qr[:,j,:]&0xF)<<(4*j)
    qz=torch.zeros((G,N//8),device=dev,dtype=torch.int32); zr=stored.reshape(G,N//8,8)
    for j in range(8): qz |= (zr[:,:,j]&0xF)<<(4*j)
    a=torch.randn((M,K),device=dev,dtype=torch.float16)
    deq=(q.float()-zero_eff[g_idx.long()].float())*scale[g_idx.long()].float()
    ref=a.float()@deq
    c=m.gptq_matmul(a,qw,scale,qz,g_idx,bits=4)
    cf=c.float()
    err=(cf-ref).abs()
    res[f"M{M}_K{K}_N{N}_g{group}"]={
      "dtype":str(c.dtype),"shape":list(c.shape),
      "max_abs":float(err.max()),"rel_fro":float(err.norm()/ref.norm()),
      "ref_absmax":float(ref.abs().max()),
      "frac_zero_out":float((cf==0).float().mean()),
      "n_nonfinite":int((~torch.isfinite(cf)).sum()),
      "quant_step_absmax":float(scale.float().max()),
    }
print(json.dumps(res))
