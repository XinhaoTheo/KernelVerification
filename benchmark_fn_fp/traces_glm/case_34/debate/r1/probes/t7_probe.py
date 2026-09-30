import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_34/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
x = torch.randn(65536, device="cuda", dtype=torch.float32)
ref = x.sum().item()

bits = []
for _ in range(300):
    out = k.blocked_sum(x)
    bits.append(out.view(torch.int32).item())

uniq = sorted(set(bits))
vals = [torch.tensor(b, dtype=torch.int32).view(torch.float32).item() for b in uniq]
print(json.dumps({
    "metric": "distinct raw output bit patterns over 300 repeated calls",
    "reason": "contract requires bitwise-identical results across calls on the same input; any >1 distinct bit pattern is a direct violation",
    "n_distinct_bits": len(uniq),
    "distinct_bits_hex": [hex(b) for b in uniq],
    "distinct_values": vals,
    "ref_sum": ref,
    "n_calls": 300,
    "N": x.numel(),
    "block": 1024,
    "dtype": "float32",
    "distribution": "standard normal (seed 0)",
}))
