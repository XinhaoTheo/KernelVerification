
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_33/kernel.py")
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)

out={}
try:
    torch.manual_seed(1)
    dev='cuda'
    M,N,K,gs,bits = 32,32,40,40,4      # ceil(K/gs)==floor(K/gs)==1 -> c1 bug inactive
    G = -(-K//gs)
    a_big = torch.randn(M,48,device=dev,dtype=torch.float32)
    a_big[:,40:] = 7.0                  # known padding so tail overread is defined memory
    a = a_big[:, :K]
    b_big = torch.randint(-2**31,2**31-1,(6,N),device=dev,dtype=torch.int32)
    b_packed = b_big[:5]                # K//8 = 5 valid rows; row 5 is the overread
    zeros_packed = torch.randint(-2**31,2**31-1,(G,N//8),device=dev,dtype=torch.int32)
    scales = torch.full((G,N),0.05,device=dev,dtype=torch.float32)

    k_idx=torch.arange(K,device=dev); n_idx=torch.arange(N,device=dev)
    q=(b_packed[(k_idx//8),:] >> ((k_idx%8)*4).unsqueeze(1)) & 15
    zq=(zeros_packed[:,(n_idx//8)] >> ((n_idx%8)*4).unsqueeze(0)) & 15
    z=(zq+1).float()
    gmap=torch.clamp(k_idx//gs,max=G-1)
    c_ref = a @ ((q.float()-z[gmap])*scales[gmap])

    c = kmod.gptq_matmul(a, b_packed, scales, zeros_packed, gs, bits=bits)
    torch.cuda.synchronize()
    err=(c-c_ref).abs().max().item(); den=c_ref.abs().max().item()
    out.update({
      "M":M,"N":N,"K":K,"group_size":gs,"BLOCK_SIZE_K":16,
      "k_iters":( -(-K//16) ),"k_covered":3*16,
      "packed_rows_valid":5,
      "max_abs_err":err,"max_abs_ref":den,"max_rel_err":err/max(den,1e-12),
      "n_mismatch_1e-3":int((c-c_ref).abs().gt(1e-3).sum().item()),
      "numel":c.numel(),
      "crashed":False,
      "metric":"max abs err of C vs reference with correct single-group mapping; only tail masking can explain error"
    })
except Exception as e:
    out.update({"crashed":True,"error":repr(e)[:400]})
print(json.dumps(out))
