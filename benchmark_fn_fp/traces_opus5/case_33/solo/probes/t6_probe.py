
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_33/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

torch.manual_seed(0)
dev = "cuda"
bits, ipb = 4, 8
maxq = 15

def build(K, N, M, gs):
    ng_ceil = -(-K // gs)
    q = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)
    # pack along K
    b_packed = torch.zeros((K // ipb, N), device=dev, dtype=torch.int32)
    for k in range(K):
        b_packed[k // ipb] |= (q[k] << ((k % ipb) * bits))
    scales = (torch.rand((ng_ceil, N), device=dev) * 0.1 + 0.01).float()
    zq = torch.randint(0, 16, (ng_ceil, N), device=dev, dtype=torch.int32)
    zeros = torch.zeros((ng_ceil, N // ipb), device=dev, dtype=torch.int32)
    for n in range(N):
        zeros[:, n // ipb] |= (zq[:, n] << ((n % ipb) * bits))
    a = torch.randn((M, K), device=dev, dtype=torch.float32)
    return q, b_packed, scales, zq, zeros, a, ng_ceil

def ref(q, scales, zq, a, K, gs, mode):
    # zero convention as in kernel: unpacked zero + 1
    z = (zq + 1).float()
    ks = torch.arange(K, device=dev)
    if mode == "ceil":
        g = ks // gs
    else:  # floor-clamp (kernel host behaviour)
        ngf = K // gs
        g = torch.clamp(ks // gs, max=ngf - 1)
    W = (q.float() - z[g]) * scales[g]
    return a @ W

out = {}
for K, gs, tag in [(64, 32, "exact"), (48, 32, "nonmultiple")]:
    N, M = 32, 32
    q, b_packed, scales, zq, zeros, a, ng = build(K, N, M, gs)
    c = kern.gptq_matmul(a, b_packed, scales, zeros, gs, bits=bits)
    r_ceil = ref(q, scales, zq, a, K, gs, "ceil")
    r_floor = ref(q, scales, zq, a, K, gs, "floor")
    denom = r_ceil.abs().max().item()
    out[tag] = dict(
        K=K, group_size=gs, rows_supplied=ng,
        max_abs_err_vs_ceil_ref=(c - r_ceil).abs().max().item(),
        max_abs_err_vs_floorclamp_ref=(c - r_floor).abs().max().item(),
        ref_absmax=denom,
        rel_err_vs_ceil=((c - r_ceil).abs().max() / max(denom,1e-9)).item(),
    )
print(json.dumps(out, indent=2))
