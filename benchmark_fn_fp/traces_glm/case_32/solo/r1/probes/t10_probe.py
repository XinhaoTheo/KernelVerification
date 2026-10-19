import torch, json
import sys
sys.path.insert(0, '/root/cases/case_33')
from kernel import gptq_matmul

torch.manual_seed(0)
dev = 'cuda'

def run_case(K, group_size, bits=4):
    M, N = 64, 64
    ipb = 32 // bits
    maxq = 2**bits - 1
    num_groups = (K + group_size - 1) // group_size  # ceil per contract
    A = torch.randn(M, K, device=dev)
    q = torch.randint(0, maxq + 1, (K, N), device=dev, dtype=torch.int64)
    packed = torch.zeros((K + ipb - 1) // ipb, N, device=dev, dtype=torch.int64)
    for j in range(ipb):
        packed |= (q[j::ipb] << (j * bits))[:packed.shape[0]]
    packed = packed.to(torch.int32)
    scales = torch.randn(num_groups, N, device=dev) * 0.1 + 0.5
    zeros = torch.randint(0, maxq, (num_groups, N), device=dev, dtype=torch.int64)
    shift = (torch.arange(N, device=dev) % ipb) * bits
    word = (torch.arange(N, device=dev) // ipb)
    zp = torch.zeros(num_groups, (N + ipb - 1) // ipb, device=dev, dtype=torch.int64)
    zp.index_add_(1, word, (zeros << shift))
    zp = zp.to(torch.int32)
    C = gptq_matmul(A, packed, scales, zp, group_size, bits)
    # reference with contract grouping (ceil groups; trailing partial group uses its own row)
    deq = torch.empty(K, N, device=dev)
    for g in range(num_groups):
        lo, hi = g * group_size, min((g + 1) * group_size, K)
        deq[lo:hi] = (q[lo:hi].float() - (zeros[g].float() + 1)) * scales[g]
    ref = A @ deq
    diff = (C - ref).abs()
    err = diff.max().item()
    ref_scale = ref.abs().max().item()
    # what error would the kernel's clamped grouping predict? (sanity: matches err)
    g_idx_kernel = torch.clamp(torch.arange(K) // group_size, max=K // group_size - 1).to(dev)
    deq_k = (q.float() - (zeros[g_idx_kernel].float() + 1)) * scales[g_idx_kernel]
    predicted = (A @ deq_k - ref).abs().max().item()
    return {'K': K, 'group_size': group_size, 'num_groups': num_groups,
            'max_abs_err': err, 'rel': err / ref_scale, 'predicted_clamped_err': predicted}

out = [run_case(80, 32), run_case(64, 32), run_case(96, 32), run_case(100, 32)]
print(json.dumps(out))
