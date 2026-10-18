
import json, sys, torch
sys.path.insert(0, "/root/cases/case_151")
import kernel

def test(x_cpu, width):
    x = x_cpu.to("cuda")
    out = kernel.run(x)
    ref = kernel.reference(x_cpu).to("cuda")
    ok = torch.equal(out, ref)
    mism = (out != ref).sum().item()
    return ok, mism

results = {}
for width in (64, 128):
    # single-coefficient unit rows: x[b,j]=1 for each j
    xs = []
    for j in range(width):
        e = torch.zeros(1, width, dtype=torch.int32); e[0, j] = 1
        xs.append(e)
    unit = torch.cat(xs, 0)
    ok_unit, m_unit = test(unit, width)
    # random batch
    g = torch.Generator().manual_seed(0)
    rnd = torch.randint(0, 3329, (16, width), generator=g, dtype=torch.int32)
    ok_rnd, m_rnd = test(rnd, width)
    # all-3328 row (extreme values)
    ex = torch.full((1, width), 3328, dtype=torch.int32)
    ok_ex, m_ex = test(ex, width)
    # constant row passes trivially check
    cs = torch.full((1, width), 1234, dtype=torch.int32)
    ok_cs, m_cs = test(cs, width)
    results[f"N{width}"] = {"unit_ok": ok_unit, "unit_mismatches": m_unit,
                            "random_ok": ok_rnd, "random_mismatches": m_rnd,
                            "extreme_ok": ok_ex, "extreme_mismatches": m_ex,
                            "constant_ok": ok_cs, "constant_mismatches": m_cs}
print(json.dumps(results))
