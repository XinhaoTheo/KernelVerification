
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_32/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
res = {}

# Case A: many duplicates, W power of 2
keys = torch.tensor([[5,5,5,5,1,1,3,3],
                     [2,2,2,2,2,2,2,2],
                     [7,1,7,1,7,1,7,1]], device='cuda', dtype=torch.int64)
sk, perm = m.triton_argsort(keys)
ref_perm = torch.argsort(keys, dim=1, stable=True)
ref_sk = torch.sort(keys, dim=1, stable=True).values
res['A'] = dict(keys=keys.tolist(), sorted=sk.tolist(), perm=perm.tolist(),
                ref_perm=ref_perm.tolist(), ref_sorted=ref_sk.tolist(),
                sorted_ok=bool(torch.equal(sk, ref_sk)),
                perm_ok=bool(torch.equal(perm, ref_perm)),
                gather_consistent=bool(torch.equal(torch.gather(keys,1,perm), sk)))

# Case B: non-power-of-2 W with duplicates
keys2 = torch.tensor([[4,4,2,2,9,9,9],
                      [1,1,1,0,0,5,5]], device='cuda', dtype=torch.int64)
sk2, perm2 = m.triton_argsort(keys2)
ref_perm2 = torch.argsort(keys2, dim=1, stable=True)
ref_sk2 = torch.sort(keys2, dim=1, stable=True).values
res['B'] = dict(keys=keys2.tolist(), sorted=sk2.tolist(), perm=perm2.tolist(),
                ref_perm=ref_perm2.tolist(), ref_sorted=ref_sk2.tolist(),
                sorted_ok=bool(torch.equal(sk2, ref_sk2)),
                perm_ok=bool(torch.equal(perm2, ref_perm2)),
                gather_consistent=bool(torch.equal(torch.gather(keys2,1,perm2), sk2)))

# Case C: random quantized keys, larger
keys3 = torch.randint(0, 8, (64, 32), device='cuda', dtype=torch.int64)
sk3, perm3 = m.triton_argsort(keys3)
ref3 = torch.sort(keys3, dim=1, stable=True)
mismatch_rows = int((perm3 != ref3.indices).any(dim=1).sum())
res['C'] = dict(shape=list(keys3.shape), sorted_ok=bool(torch.equal(sk3, ref3.values)),
                perm_ok=bool(torch.equal(perm3, ref3.indices)),
                rows_with_perm_mismatch=mismatch_rows, total_rows=64,
                gather_consistent=bool(torch.equal(torch.gather(keys3,1,perm3), sk3)),
                nondecreasing=bool((sk3[:,1:]>=sk3[:,:-1]).all()))

# Case D: determinism across repeats
reps = [m.triton_argsort(keys3)[1] for _ in range(3)]
res['D'] = dict(deterministic=all(torch.equal(reps[0], r) for r in reps))
print(json.dumps(res))
