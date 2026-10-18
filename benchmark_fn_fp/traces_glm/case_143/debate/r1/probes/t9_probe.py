
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_143/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
res = {}
for N in (32, 64, 128):
    g = torch.Generator(device="cpu").manual_seed(1234 + N)
    x = torch.randint(-32768, 32768, (5, N), generator=g, dtype=torch.int32).to("cuda", torch.int16)
    out = k.run(x); ref = k.reference(x)
    exact = torch.equal(out, ref)
    # how many coefficient positions match (identity permutation check)
    match_pos = int((out == ref).float().mean(dim=0).gt(0.9999).sum()) if not exact else N
    # is out a row-permutation of ref?
    perm_ok = any(torch.equal(out, ref[:, p]) for p in [None])
    res[f"N{N}"] = {"exact": exact, "rows_fully_matching_ref_positions": match_pos}
print(json.dumps(res))
