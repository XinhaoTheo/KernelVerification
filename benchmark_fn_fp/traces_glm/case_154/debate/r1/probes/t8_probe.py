import torch, json, sys
sys.path.insert(0, "/root/cases/case_154")
import kernel as K

dev = "cuda"
rows = []
# Catastrophic cancellation cases with large-magnitude terms
# (a*c) ~ -(b*d): e.g. a=65504,b=65504,c=65504,d=-65504 -> a*c=4.29e9, b*d=-4.29e9
cands = [
    (65504.0, 65504.0, 65504.0, -65504.0),
    (65504.0, 1.0, 65504.0, -1.0),        # a*c = 65504, b*d = -1 -> real num = 65503
    (-1.0, 65504.0, 65504.0, 1.0),        # b*c=4.29e9, a*d=-1 -> imag num ~4.29e9
    (65504.0, -65504.0, 65504.0, -65504.0),  # imag num: b*c - a*d = -4.29e9 - (-4.29e9) = 0 exact cancel
    (30000.0, 30000.0, 30000.0, -30000.0),
    (65504.0, 65504.0, 30000.0, -30000.0),   # a*c=1.965e9, b*d=-1.965e9
    (1000.0, 1000.0, 1000.0, -1000.0),
]
for a,b,c,d in cands:
    rows.append([a,b,c,d])

num = torch.tensor([[r[0],r[1]] for r in rows], dtype=torch.float16, device=dev)
den = torch.tensor([[r[2],r[3]] for r in rows], dtype=torch.float16, device=dev)
try:
    out = K.run(num, den)
    ref = K.reference(num, den)
    tol = 1e-7 + 5e-7*ref.abs()
    err = (out.double()-ref.double()).abs()
    fail = err > tol
    print(json.dumps({
        "n_rows": len(rows),
        "rows": rows,
        "out": out.cpu().tolist(),
        "ref": ref.cpu().tolist(),
        "err": err.cpu().tolist(),
        "tol": tol.cpu().tolist(),
        "fail": fail.cpu().tolist(),
        "any_fail": bool(fail.any())
    }))
except Exception as e:
    print(json.dumps({"error": str(e)}))
