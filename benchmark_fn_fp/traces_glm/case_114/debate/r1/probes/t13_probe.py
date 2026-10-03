
import torch, json, math, sys
sys.path.insert(0, "/root/cases/case_114")
import kernel as K

torch.manual_seed(0)
dev = "cuda"
# B=1, Q=1, Hq=Hk=1, D=64, S=16, P=8, length=100, window=5
B, Q, HQ, D = 1, 1, 1, 64
S, P, HK = 16, 8, 1
length = 100
window = 5
q = (torch.rand((B, Q, HQ, D), device=dev, generator=torch.Generator(device=dev).manual_seed(1)) * 2 - 1).half()
k = (torch.rand((P, S, HK, D), device=dev, generator=torch.Generator(device=dev).manual_seed(2)) * 2 - 1).half()
v = (torch.rand(k.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(3)) * 2 - 1).half()
# unique page table covering positions 0..99 (7 pages)
table = torch.arange(0, 7, device=dev, dtype=torch.int32).unsqueeze(0)
lengths = torch.tensor([length], dtype=torch.int32, device=dev)

# context = 99. Kernel tile bounds for qb=0: first_key=99-5+1=95 -> tile_start=1; last_key=99 -> tile_end=2.
# Tile 1 covers logical 64..127; loaded (logical<100) positions 64..99 include expired 64..94.
# Poison physical slot of logical position 64: page 4, slot 0 (unique mapping, no allowed position uses it).
lo = max(0, length - Q - window + 1)
assert lo == 95
v[4, 0, :, :] = float("nan")

K.validate_inputs(q, k, v, table, lengths, window)  # must pass: only t>=95 checked
out = K.run(q, k, v, table, lengths, window)
torch.cuda.synchronize()
nan_count = int(torch.isnan(out.float()).sum())
ref = K.reference(q, k, v, table, lengths, window)
ref_nan = int(torch.isnan(ref).sum())
ratio = K.error_ratio(out, (q, k, v, table, lengths, window))
print(json.dumps({
    "claim": "c1", "window": window, "length": length, "poisoned_logical_pos": 64,
    "poisoned_slot": [4, 0], "lo": lo,
    "output_nan_count": nan_count, "total_output_elements": out.numel(),
    "reference_nan_count": ref_nan, "error_ratio": ratio,
    "finite_output_violation": nan_count > 0,
}))
