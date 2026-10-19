import sys, json, torch
sys.path.insert(0, "/root/cases/case_156")
import kernel as K
dev = "cuda"
m_max = (1 << 31) - 1
# Build valid int32 a values via int64 then cast, avoiding tensor overflow errors.
# Target R in 250..400 (spanning the [-255,255] preclamp boundary): R = a*m/2^(31+r) ~ a (r=0, m~2^31).
a_list = list(range(240, 401, 2)) + [-x for x in range(240, 401, 2)]
extremes = [-(1 << 31), (1 << 31) - 1, -(1 << 31) + 1, (1 << 31) - 2]
a_list = a_list + extremes
a_t = torch.tensor(a_list, dtype=torch.int64)
assert ((a_t >= -(1 << 31)) & (a_t <= (1 << 31) - 1)).all()
acc = a_t.to(torch.int32).to(dev).reshape(len(a_list), 1)
C = 1
mul = torch.full((C,), m_max, dtype=torch.int32, device=dev)
res = {}
for z in [-128, 127, 0, 50]:
    for r in [0, 1, 2]:
        rsh = torch.full((C,), r, dtype=torch.int32, device=dev)
        zp = torch.full((C,), z, dtype=torch.int32, device=dev)
        out = K.run(acc, mul, rsh, zp)
        ref = K.reference(acc, mul, rsh, zp)
        mism = (out != ref).sum().item()
        bad = (out != ref).nonzero().flatten().tolist()
        detail = []
        for i in bad[:5]:
            a = a_list[i]
            R = (a * m_max + (1 << (30 + r))) >> (31 + r)
            detail.append({"a": a, "R": R, "z": z, "kernel": out[i, 0].item(), "ref": ref[i, 0].item()})
        res[f"z{z}_r{r}"] = {"mismatches": mism, "n": len(a_list), "mismatch_detail": detail}
allm = sum(v["mismatches"] for v in res.values())
print(json.dumps({"metric": "exact_int8_equality_preclamp_band_and_extremes",
  "a_range": "240..400 step2 + negatives + int32 extremes", "m": m_max,
  "total_mismatches": allm, "per_config": res}))