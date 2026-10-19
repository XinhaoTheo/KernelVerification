import torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_158/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(1234)
results = {}
worst = None
for N in range(9, 33):
    B = 3
    # distinguishing: encode step index in bias; matrices mixed incl. singular and all-ones
    matrix = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8)
    matrix[0] = 1  # all ones row
    matrix[1] = torch.tensor([1,0,0,0], dtype=torch.int8).view(1,1,1,4).expand(1,N,2,2)
    bias = torch.randint(-30000, 30000, (B, N, 2), generator=g, dtype=torch.int16)
    initial = torch.randint(-32768, 32767, (B, 2), generator=g, dtype=torch.int16)
    m, b, i = matrix.to(dev), bias.to(dev), initial.to(dev)
    try:
        out = k.run(m, b, i)
        ref = k.reference(m, b, i)
        ok = bool(torch.equal(out, ref))
        results[str(N)] = ok
        if not ok and worst is None:
            diff = (out - ref).abs().max().item()
            worst = {"N": N, "max_abs_diff": diff}
    except Exception as e:
        results[str(N)] = f"error: {e}"
print({"per_N_exact": results, "first_mismatch": worst, "claim": "c2", "metric": "bitwise exact int64 equality vs reference for N=9..32 (chunks 2..4, padded C=4) with adversarial singular/all-ones/random matrices and full-range int16 bias/initial"})