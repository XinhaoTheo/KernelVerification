import sys, torch, json
sys.path.insert(0, "/root/cases/case_110")
import kernel

dev = "cuda" if torch.cuda.is_available() else "cpu"
B, L, D, K = 2, 64, 8, 3
g = torch.Generator().manual_seed(0)
state = torch.randn((B, L, D), generator=g).to(dev)
events = torch.randn((B, K, D), generator=g).to(dev)
NEG = -2147483648  # int32 bit pattern 0x80000000 == -0.0
i32 = state.view(torch.int32)
i32[0, 5, 2] = NEG    # -0.0 in state-branch region (row < L-K)
i32[0, L-1, 1] = NEG  # -0.0 in events-branch region
i32[0, 6, 0] = 0      # +0.0 in state-branch region
e32 = events.view(torch.int32)
e32[1, 0, 3] = NEG
e32[1, K-1, 4] = NEG
e32[0, K-1, 7] = 0

ref = kernel.reference(state, events)
expected_bits = ref.view(torch.int32).clone()
kernel.run(state, events)
torch.cuda.synchronize()
got_bits = state.view(torch.int32)
mismatches = int((got_bits != expected_bits).sum().item())
neg_positions = (expected_bits == NEG)
neg_preserved = int(((got_bits == NEG) & neg_positions).sum().item())
neg_zero_ref = int(neg_positions.sum().item())
result = {
    "device": str(state.device),
    "bit_mismatches_total": mismatches,
    "neg_zero_count_ref": neg_zero_ref,
    "neg_zero_preserved": neg_preserved,
    "plus_zero_count_ref": int((expected_bits == 0).sum().item()),
    "events_preserved_bits": bool((events.view(torch.int32) == e32).all().item()),
}
print(json.dumps(result))
