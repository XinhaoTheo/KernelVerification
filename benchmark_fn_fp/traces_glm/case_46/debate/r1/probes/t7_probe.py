import torch, json, sys
sys.path.insert(0, "/root/cases/case_119")
import kernel

dev = "cuda"
torch.manual_seed(0)

# Target: 5 expert blocks total (num_pid_m=5, GROUP_M=4 -> last group size 1).
# Use E=5 experts, Q=1, M=80 tokens. Assign 16 tokens to expert 0, and 16 each
# to experts 1..3, 16 to expert 4 -> exactly 5 blocks of 16 routes each.
# ids must be int32 [M,Q], distinct per token (trivial for Q=1).
M, K, N, E, Q = 80, 32, 96, 5, 1
counts = [16, 16, 16, 16, 16]
assert sum(counts) == M
ids_list = []
for e, c in enumerate(counts):
    ids_list += [e] * c
ids = torch.tensor(ids_list, dtype=torch.int32, device=dev).reshape(M, Q)
gen = torch.Generator(device="cpu").manual_seed(7)
x = (torch.rand((M, K), generator=gen) - .5).half().to(dev)
W = (torch.rand((E, K, N), generator=gen) - .5).half().to(dev)
w = (torch.rand((M, Q), generator=gen) * 0.5).float().to(dev)

# check legality
info = kernel.validate_inputs(x, W, ids, w)

# expected block structure from _prepare_routes
sorted_ids, expert_blocks = kernel._prepare_routes(ids, E, 16)
num_pid_m = sorted_ids.numel() // 16

out = kernel.run(x, W, ids, w)
ref = kernel.reference(x, W, ids, w)

diff = (out.double() - ref).abs()
tol = 0.01 + 0.0002 * ref.abs()
viol = diff > tol
# which experts' tokens are affected: column ranges
n_viol = int(viol.sum())
viol_cols = [int(j) for j in viol.any(dim=0).nonzero().flatten().tolist()]
print(json.dumps({
    "info": info, "num_blocks": int(expert_blocks.numel()),
    "num_pid_m": num_pid_m, "M": M, "K": K, "N": N, "E": E, "Q": Q,
    "max_abs_diff": float(diff.max()), "n_violations": n_viol,
    "violating_cols": viol_cols[:20], "n_violating_cols": len(viol_cols),
    "tolerance": 0.01,
    "expected_n_tile_cols": (N + 31) // 32,
}))
