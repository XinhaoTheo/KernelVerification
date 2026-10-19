import json, torch
import kernel as K

dev = "cuda" if torch.cuda.is_available() else "cpu"
base = (2.0 * torch.randn((1000,), dtype=torch.float16)).to(dev)
x = base[300:]  # contiguous view, storage offset 300, size 700
assert x.storage_offset() == 300 and x.is_contiguous()
rejected = False
msg = None
try:
    out = K.run(x)
except Exception as e:
    rejected = True
    out = None
    msg = str(e)

res = {"rejected": rejected, "error_msg": msg, "dev": dev}
if not rejected:
    ref = K.reference(x).to(torch.float64)
    err = (out.to(torch.float64) - ref).abs()
    full = K.run(base)
    res.update({
        "n": int(x.numel()),
        "max_abs_err": float(err.max()),
        "tol": 1e-6,
        "out_fp32_fresh": (out.dtype == torch.float32 and out.is_cuda),
        "view_matches_fullslice_bitwise": bool(torch.equal(out, full[300:])),
        "input_preserved": bool(torch.equal(x, base[300:])),
    })
print(json.dumps(res))
