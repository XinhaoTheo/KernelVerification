
import json, torch
import kernel as K

dev = "cuda" if torch.cuda.is_available() else "cpu"
base = (2.0 * torch.randn((1000,), dtype=torch.float16)).to(dev)
x = base[300:]  # contiguous view, storage offset 300, size 700
assert x.storage_offset() == 300 and x.is_contiguous()
rejected = False
try:
    out = K.run(x)
except Exception as e:
    rejected = True
    out = None
    msg = str(e)
ref = K.reference(x).to(torch.float64).to(dev) if not rejected else None
err = (out.to(torch.float64) - ref).abs() if out is not None else None
print(json.dumps({
    "rejected": rejected,
    "error_msg": msg if rejected else None,
    "n": 700 if not rejected else None,
    "max_abs_err": float(err.max()) if err is not None else None,
    "tol": 1e-6,
    "view_matches_full": bool(torch.allclose(out, K.run(base)[300:].to(torch.float64), atol=0, rtol=0)) if out is not None else None,
    "input_preserved": bool(torch.equal(x, base[300:])) if out is not None else None,
    "dev": dev,
}))
