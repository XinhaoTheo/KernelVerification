
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_143/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
res = {}
for N in (32, 64, 128):
    g = torch.Generator(device="cpu").manual_seed(999 + N)
    x = torch.randint(-32768, 32768, (3, N), generator=g, dtype=torch.int32).to("cuda", torch.int16)
    out = k.run(x); ref = k.reference(x)
    # diagnostic: is out any permutation of ref's rows (row-set equality)?
    rows_ref = sorted(map(tuple, ref.tolist())); rows_out = sorted(map(tuple, out.tolist()))
    row_set_equal = rows_ref == rows_out
    # duplicate coefficient rows in out (sign of broken gather)?
    dup_rows = len(rows_out) != len(set(rows_out))
    # correctness per width
    exact = torch.equal(out, ref)
    res[f"N{N}"] = {"exact": exact, "rows_are_permutation_of_ref_rows": row_set_equal, "duplicate_rows_in_out": dup_rows}
print(json.dumps(res))
