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
    q = torch.randint(0, maxq + 1, (K, N), device=dev, dtype=torch.int32)
    packed = torch.zeros((K + ipb - 1) // ipb, N, device=dev, dtype=torch.int32)
    for j in range(ipb):
        packed |= (q[j::ipb] << (j * bits))[:packed.shape[0]]
    scales = torch.randn(num_groups, N, device=dev) * 0.1 + 0.5
    zeros = torch.randint(0, maxq, (num_groups, N), device=dev, dtype=torch.int32)
    # pack zeros per contract: zeros tensor is (num_groups, N) ints; kernel indexes zeros_ptr + g_idx*stride_zeros + bn//ipb
    # So zeros must be packed along N into int32 words: shape (num_groups, ceil(N/ipb))
    zp = torch.zeros(num_groups, (N + ipb - 1) // ipb, device=dev, dtype=torch.int32)
    for j in range(ipb):
        zp[:, j::ipb] |= (zeros[:, j::ipb] << (j * bits))[:, :zp.shape[1]]
    C = gptq_matmul(A, packed, scales, zp, group_size, bits)
    # reference
    deq = torch.empty(K, N, device=dev)
    for k in range(K):
        g = k // group_size
        deq[k] = (q[k].float() - (zeros[g].float() + 1)) * scales[g]
    ref = A @ deq
    err = (C - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    return {'K': K, 'group_size': group_size, 'num_groups': num_groups, 'max_abs_err': err, 'rel': rel}

out = []
# partial trailing group case
out.append(run_case(80, 32))
# control: divisible case
out.append(run_case(64, 32))
print(json.dumps(out))
