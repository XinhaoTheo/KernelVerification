
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_159/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator("cpu").manual_seed(7)
results = {}
for N in range(1, 33):
    B = 5
    matrix = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8).to(dev)
    bias = torch.randint(-32768, 32768, (B, N, 2), generator=g, dtype=torch.int16).to(dev)
    initial = torch.randint(-32768, 32768, (B, 2), generator=g, dtype=torch.int16).to(dev)
    try:
        out = k.run(matrix, bias, initial)
        ref = k.reference(matrix, bias, initial)
        results[N] = {"exact": bool(torch.equal(out, ref)),
                      "mismatch": int((out != ref).sum())}
    except Exception as ex:
        results[N] = {"error": f"{type(ex).__name__}: {ex}"}
bad = {n: r for n, r in results.items() if r.get("exact") is not True}
print(json.dumps({"claim": "c2", "ns_tested": 32, "num_bad_N": len(bad), "bad_N": bad,
                  "n1_8_exact": all(results[n].get("exact") for n in range(1,9)),
                  "n9_16_exact": all(results[n].get("exact") for n in range(9,17)),
                  "n17_32_exact": all(results[n].get("exact") for n in range(17,33))}))
