
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_33/kernel.py")
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)

out={}
try:
    torch.manual_seed(2)
    dev='cuda'
    M,N,K,gs,bits = 32,32,16,32,4      # K < group_size -> num_groups = 0
    G = -(-K//gs)                       # contract supplies 1 row
    a = torch.randn(M,K,device=dev,dtype=torch.float32)
    b_packed = torch.randint(-2**31,2**31-1,(K//8,N),device=dev,dtype=torch.int32)
    # scales/zeros are views into row 1 of a 3-row buffer so a row -1 read is defined memory
    scales_big = torch.stack([torch.full((N,),0.9,device=dev),
                              torch.full((N,),0.02,device=dev),
                              torch.full((N,),0.3,device=dev)]).contiguous()
    zeros_big = torch.randint(-2**31,2**31-1,(3,N//8),device=dev,dtype=torch.int32)
    scales = scales_big[1:2]
    zeros_packed = zeros_big[1:2]

    k_idx=torch.arange(K,device=dev); n_idx=torch.arange(N,device=dev)
    q=(b_packed[(k_idx//8),:] >> ((k_idx%8)*4).unsqueeze(1)) & 15
    def deq(scale_row, zero_row):
        zq=(zero_row[(n_idx//8)] >> ((n_idx%8)*4)) & 15
        return (q.float()-(zq+1).float().unsqueeze(0))*scale_row.unsqueeze(0)
    c_ref  = a @ deq(scales_big[1], zeros_big[1])   # correct: group 0 == supplied row
    c_prev = a @ deq(scales_big[0], zeros_big[0])   # what row -1 would give

    g_idx_min = int(torch.clamp(torch.arange(K,device=dev,dtype=torch.int32)//gs, max=(K//gs)-1).min().item())
    c = kmod.gptq_matmul(a, b_packed, scales, zeros_packed, gs, bits=bits)
    torch.cuda.synchronize()
    e_ref=(c-c_ref).abs().max().item(); e_prev=(c-c_prev).abs().max().item()
    out.update({"M":M,"N":N,"K":K,"group_size":gs,"ceil_groups":G,"num_groups_wrapper":K//gs,
      "kernel_g_idx_value":g_idx_min,
      "max_abs_err_vs_correct_row0":e_ref,"max_abs_err_vs_row_minus_1":e_prev,
      "max_abs_ref":c_ref.abs().max().item(),
      "matches_correct":bool(e_ref<1e-3),"matches_row_minus_1":bool(e_prev<1e-3),
      "crashed":False,
      "metric":"max abs err of C against dequant using supplied row 0 vs the row physically preceding scales/zeros"})
except Exception as e:
    out.update({"crashed":True,"error":repr(e)[:400]})
print(json.dumps(out))
