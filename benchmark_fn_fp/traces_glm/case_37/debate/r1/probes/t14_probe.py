import sys, torch, json
sys.path.insert(0, "/root/cases/case_110")
import kernel

dev = "cuda" if torch.cuda.is_available() else "cpu"
B, L, D, K = 2, 64, 8, 3
g = torch.Generator().manual_seed(0)
state = torch.randn((B, L, D), generator=g).to(dev)
events = torch.randn((B, K, D), generator=g).to(dev)
# sprinkle signed zeros in both branches
i32 = state.view(torch.int32)
i32[0, 5, 2] = 0x80000000   # -0.0 in state-branch region
i32[0, L-1, 1] = 0x80000000 # -0.0 in events-branch region (row L-1 >= L-K)
e32 = events.view(torch.int32)
e32[1, 0, 3] = 0x80000000
e32[1, K-1, 4] = 0x80000000
# also +0.0
i32[0, 6, 0] = 0
e32[0, K-1, 7] = 0

ref = kernel.reference(state, events)
expected_bits = ref.view(torch.int32).clone()
out = kernel.run(state, events)
torch.cuda.synchronize()
got_bits = state.view(torch.int32)
mismatches = (got_bits != expected_bits).sum().item()
neg_zero_ref = (expected_bits == 0x80000000).sum().item()
neg_zero_got = (got_bits == 0x80000000).sum().item()
# specifically check where ref has -0.0
neg_positions = (expected_bits == 0x80000000)
neg_preserved = ((got_bits == 0x80000000) & neg_positions).sum().item()
result = {
    "device": str(state.device),
    "shapes": [list(state.shape), list(events.shape)],
    "bit_mismatches_total": mismatches,
    "neg_zero_count_ref": neg_zero_ref,
    "neg_zero_count_got": neg_zero_got,
    "neg_zero_preserved": neg_preserved,
    "events_preserved_bits": bool((events.view(torch.int32) == e32).all().item()),
    "max_abs_err": (state.float() - ref.float()).abs().max().item(),
}
print(json.dumps(result))
