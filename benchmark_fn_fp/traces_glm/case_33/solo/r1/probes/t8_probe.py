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
    # pack zeros along N: word n//ipb holds col n at shift (n%ipb)*bits
    shift = (torch.arange(N, device=dev) % ipb) * bits
    word = (torch.arange(N, device=dev) // ipb)
    zp = torch.zeros(num_groups, (N + ipb - 1) // ipb, device=dev, dtype=torch.int64)
    zp.index_add_(1, word, (zeros << shift))
    zp = zp.to(torch.int32)
    C = gptq_matmul(A, packed, scales, zp, group_size, bits)
    # reference: group_of(k) = k // group_size, using ceil rows
    deq = torch.empty(K, N, device=dev)
    for g in range(num_groups):
        lo, hi = g * group_size, min((g + 1) * group_size, K)
        deq[lo:hi] = (q[lo:hi].float() - (zeros[g].float() + 1)) * scales[g]
    ref = A @ deq
    err = (C - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    # also: per-column error attribution for trailing group
    tail = group_size * (K // group_size)  # start of partial group
    tail_err = (C - ref)[:, tail:].abs().max().item() if tail < K else 0.0
    head_err = (C - ref)[:, :tail].abs().max().item() if tail > 0 else 0.0
    return {'K': K, 'group_size': group_size, 'num_groups': num_groups,
            'max_abs_err': err, 'rel': rel, 'head_err': head_err, 'tail_err': tail_err}

out = []
out.append(run_case(80, 32))   # partial trailing group (contract-critical)
out.append(run_case(64, 32))   # control: K divisible by group_size
out.append(run_case(96, 32))   # control: divisible, ceil == floor
print(json.dumps(out))
