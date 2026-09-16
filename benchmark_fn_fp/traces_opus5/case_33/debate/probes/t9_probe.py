
import json, importlib.util, torch, math
spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_33/kernel.py")
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)

torch.manual_seed(0)
dev='cuda'
M,N,K,gs,bits = 32,32,48,32,4
G = -(-K//gs)   # ceil = 2
a = torch.randn(M,K,device=dev,dtype=torch.float32)
b_packed = torch.randint(-2**31, 2**31-1, (K//8, N), device=dev, dtype=torch.int32)
zeros_packed = torch.randint(-2**31, 2**31-1, (G, N//8), device=dev, dtype=torch.int32)
# make the trailing (short) group's scale row very different from group 0
scales = torch.stack([torch.full((N,),0.01,device=dev), torch.full((N,),0.5,device=dev)]).contiguous()

k_idx = torch.arange(K, device=dev)
n_idx = torch.arange(N, device=dev)
q = (b_packed[(k_idx//8), :] >> ((k_idx%8)*4).unsqueeze(1)) & 15
zq = (zeros_packed[:, (n_idx//8)] >> ((n_idx%8)*4).unsqueeze(0)) & 15
z = (zq + 1).float()

def deq(gmap):
    return (q.float() - z[gmap]) * scales[gmap]

gmap_true  = torch.clamp(k_idx//gs, max=G-1)                 # 0..31 ->0, 32..47 ->1
gmap_wrong = torch.clamp(k_idx//gs, max=(K//gs)-1)           # kernel's floor model -> all 0
c_ref   = a @ deq(gmap_true)
c_wrong = a @ deq(gmap_wrong)

c = kmod.gptq_matmul(a, b_packed, scales, zeros_packed, gs, bits=bits)
torch.cuda.synchronize()

err_ref   = (c-c_ref).abs().max().item()
err_wrong = (c-c_wrong).abs().max().item()
den = c_ref.abs().max().item()
print(json.dumps({
 "M":M,"N":N,"K":K,"group_size":gs,"ceil_groups":G,"floor_groups":K//gs,
 "g_idx_unique_kernel":sorted(set(gmap_wrong.tolist())),
 "g_idx_unique_contract":sorted(set(gmap_true.tolist())),
 "max_abs_err_vs_contract_ref":err_ref,
 "max_abs_err_vs_floor_model":err_wrong,
 "max_abs_ref":den,
 "max_rel_err_vs_contract_ref":err_ref/max(den,1e-12),
 "matches_floor_model":bool(err_wrong<1e-3),
 "matches_contract_ref":bool(err_ref<1e-3),
 "metric":"elementwise max abs err of C vs two candidate group-mapping references"
}))
