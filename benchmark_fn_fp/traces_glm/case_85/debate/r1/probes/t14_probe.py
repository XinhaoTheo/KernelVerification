import torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_158/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(1234)
results = {}
worst = None
sing = torch.tensor([1,0,0,1], dtype=torch.int8)  # singular? use diag(1,0)
sing = torch.tensor([1,0,0,0], dtype=torch.int8)
for N in range(9, 33):
    B = 3
    matrix = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8)
    matrix[0] = 1  # all ones
    # singular diag(1,0) repeated
    matrix[1] = sing.view(1, 1, 2, 2).expand(1, N, 2, 2).contiguous()
    bias = torch.randint(-30000, 30000, (B, N, 2), generator=g, dtype=torch.int16)
    initial = torch.randint(-32768, 32767, (B, 2), generator=g, dtype=torch.int16)
    m, b, i = matrix.to(dev), bias.to(dev), initial.to(dev)
    try:
        out = k.run(m, b, i)
        ref = k.reference(m, b, i)
        ok = bool(torch.equal(out, ref))
        results[str(N)] = ok
        if not ok and worst is None:
            worst = {"N": N, "max_abs_diff": (out - ref).abs().max().item()}
    except Exception as e:
        results[str(N)] = f"error: {e}"
print({"per_N_exact": results, "first_mismatch": worst, "claim": "c2", "metric": "bitwise exact int64 equality vs reference, N=9..32, adversarial singular/all-ones/random matrices, full-range int16 bias/initial"})