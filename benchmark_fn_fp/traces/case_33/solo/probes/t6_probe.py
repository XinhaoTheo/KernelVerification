
import torch, json, sys, importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_33/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

torch.manual_seed(0)
dev = "cuda"

def make(M, N, K, gs, bits=4):
    ipb = 32 // bits
    ngroups_ceil = -(-K // gs)
    a = torch.randn(M, K, device=dev, dtype=torch.float32)
    q = torch.randint(0, 2**bits, (K, N), device=dev, dtype=torch.int32)
    # pack along K
    b_packed = torch.zeros(K // ipb, N, device=dev, dtype=torch.int32)
    for k in range(K):
        b_packed[k // ipb] |= (q[k] << (bits * (k % ipb)))
    scales = (torch.rand(ngroups_ceil, N, device=dev, dtype=torch.float32) * 0.5 + 0.05)
    zq = torch.randint(0, 2**bits, (ngroups_ceil, N), device=dev, dtype=torch.int32)
    zeros_packed = torch.zeros(ngroups_ceil, N // ipb, device=dev, dtype=torch.int32)
    for n in range(N):
        zeros_packed[:, n // ipb] |= (zq[:, n] << (bits * (n % ipb)))
    return a, q, b_packed, scales, zq, zeros_packed, ngroups_ceil

def ref(a, q, scales, zq, gs):
    K = a.shape[1]
    kidx = torch.arange(K, device=dev)
    g = kidx // gs                      # contract: group_of(k) = k // group_size
    deq = (q.float() - (zq[g].float() + 1)) * scales[g]
    return a @ deq, g

out = {}
for tag, (M, N, K, gs) in {
    "control_K64_gs32": (32, 32, 64, 32),
    "partial_K80_gs32": (32, 32, 80, 32),
    "partial_K48_gs32": (32, 32, 48, 32),
}.items():
    a, q, b_packed, scales, zq, zeros_packed, ng = make(M, N, K, gs)
    c = kern.gptq_matmul(a, b_packed, scales, zeros_packed, gs, bits=4)
    r, g = ref(a, q, scales, zq, gs)
    err = (c - r).abs()
    denom = r.abs().clamp_min(1e-6)
    # what group indices the kernel actually used
    num_groups_floor = K // gs
    g_kernel = torch.clamp(torch.arange(K, device=dev) // gs, max=num_groups_floor - 1)
    out[tag] = dict(
        K=K, gs=gs, ceil_groups=ng, kernel_num_groups=num_groups_floor,
        max_abs_err=err.max().item(), max_rel_err=(err/denom).max().item(),
        ref_absmax=r.abs().max().item(),
        g_contract_tail=g[-8:].tolist(), g_kernel_tail=g_kernel[-8:].tolist(),
        allclose=torch.allclose(c, r, atol=1e-3, rtol=1e-3),
    )

# also verify kernel matches the *floor/clamp* semantics (i.e. bug is exactly the group mapping)
a, q, b_packed, scales, zq, zeros_packed, ng = make(32, 32, 80, 32)
c = kern.gptq_matmul(a, b_packed, scales, zeros_packed, 32, bits=4)
gk = torch.clamp(torch.arange(80, device=dev) // 32, max=80 // 32 - 1)
deq_k = (q.float() - (zq[gk].float() + 1)) * scales[gk]
r_kernelsem = a @ deq_k
out["kernel_matches_clamped_semantics"] = dict(
    max_abs_err=(c - r_kernelsem).abs().max().item(),
    allclose=torch.allclose(c, r_kernelsem, atol=1e-3, rtol=1e-3),
)
print(json.dumps(out, indent=1))
