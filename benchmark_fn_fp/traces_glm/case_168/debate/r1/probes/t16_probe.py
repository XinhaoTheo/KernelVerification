import torch, sys, json
sys.path.insert(0, '/root/cases/case_168')
from kernel import run

# kernel's own per-byte map (from the 256-sweep), used to isolate tail masking
sweep = torch.arange(256, dtype=torch.uint8, device='cuda')
kmap = run(sweep).cpu().tolist()

results = {}
for N in [1, 2, 255, 256, 257, 1027, 1028, 131072]:
    x = torch.randint(0, 256, (N,), dtype=torch.uint8, device='cuda')
    out = run(x).cpu().tolist()
    xs = x.cpu().tolist()
    conform = all(out[i] == kmap[xs[i]] for i in range(N))
    if N % 256 != 0:
        start = (N // 256) * 256
        tail_conform = all(out[i] == kmap[xs[i]] for i in range(start, N))
    else:
        tail_conform = True
    results[str(N)] = {"conform_everywhere": bool(conform), "tail_conform": bool(tail_conform)}

# also verify the kernel writes only N bytes (no OOB into a sentinel output is not directly
# observable; instead check input buffer bytes beyond the tensor are untouched via a padded view)
pad = torch.randint(0, 256, (300,), dtype=torch.uint8, device='cuda')
pad_clone = pad.clone()
view = pad[2:258]  # N=256, offset 2, bytes before/after must remain intact
run(view)
surround_preserved = torch.equal(pad[:2], pad_clone[:2]) and torch.equal(pad[258:], pad_clone[258:])

print(json.dumps({"metric": "per-position conformance to kernel's own byte map at all lengths incl. partial tails + surrounding-byte preservation",
                  "results": results, "surrounding_bytes_preserved": bool(surround_preserved)}))