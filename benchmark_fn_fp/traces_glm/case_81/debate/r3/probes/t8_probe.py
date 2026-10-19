import torch, sys
sys.path.insert(0, "/root/cases/case_154")
import kernel as K

cases = [
    [1000.0, 1.0, -1.0, 1000.0],
    [1.0, 1000.0, 1000.0, 1.0],
    [65504.0, 1.0, -1.0, 65504.0],
    [2048.0, 3.0, -3.0, 2048.0],
    [100.0, 0.5, -0.5, 100.0],
]
num = torch.tensor([[a, b] for a, b, c, d in cases], dtype=torch.float16, device="cuda")
den = torch.tensor([[c, d] for a, b, c, d in cases], dtype=torch.float16, device="cuda")
out = K.run(num, den).double()
ref = K.reference(num.cpu(), den.cpu()).double().cuda()
err = (out - ref).abs()
tol = 1e-7 + 5e-7 * ref.abs()
ok = err <= tol
print("ref:", ref.tolist())
print("out:", out.tolist())
print("err:", err.tolist())
print("tol:", tol.tolist())
print("ok:", ok.tolist())
import json
print(json.dumps({"n_violations": int((~ok).sum()), "err": err.tolist(), "tol": tol.tolist(), "ok": ok.tolist()}))