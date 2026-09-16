
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_33/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev='cuda'
torch.manual_seed(0)

def ref(a, bp, scales, zeros, group_size, bits=4):
    M,K = a.shape; N = bp.shape[1]
    ks = torch.arange(K, device=dev)
    q = (bp[(ks//8), :] >> ((ks%8)*4).unsqueeze(1)) & 15
    ns = torch.arange(N, device=dev)
    zp = ((zeros[:, ns//8] >> ((ns%8)*4).unsqueeze(0)) & 15) + 1
    g = ks//group_size
    deq = (q.float() - zp[g].float()) * scales[g].float()
    return a @ deq, g

def build(M,N,K,group_size):
    G = -(-K//group_size)
    a = torch.randn(M,K, device=dev, dtype=torch.float32)
    bp = torch.randint(0, 2**31-1, (K//8, N), device=dev, dtype=torch.int32)
    scales = (torch.rand(G,N, device=dev, dtype=torch.float32)*0.1+0.01)
    zeros = torch.randint(0, 2**31-1, (G, N//8), device=dev, dtype=torch.int32)
    return a,bp,scales,zeros,G

out={}
cases = {"exact_K64_gs32":(32,32,64,32), "short_K48_gs32":(32,32,48,32)}
for name,(M,N,K,gs) in cases.items():
    a,bp,scales,zeros,G = build(M,N,K,gs)
    r,gref = ref(a,bp,scales,zeros,gs)
    c = k.gptq_matmul(a,bp,scales,zeros,gs,4)
    err = (c-r).abs()
    den = r.abs().clamp_min(1e-6)
    rel = err/den
    ng = K//gs
    gk = torch.clamp(torch.arange(K, device=dev)//gs, max=ng-1)
    out[name]=dict(shape=[M,N,K], group_size=gs, rows_supplied=G,
                   kernel_num_groups=ng,
                   g_idx_ref_tail=gref[-4:].tolist(),
                   g_idx_kernel_tail=gk[-4:].tolist(),
                   max_abs_err=float(err.max()),
                   max_rel_err=float(rel.max()),
                   ref_absmax=float(r.abs().max()),
                   frac_rel_gt_1em2=float((rel>1e-2).float().mean()))
print(json.dumps(out, indent=2))
