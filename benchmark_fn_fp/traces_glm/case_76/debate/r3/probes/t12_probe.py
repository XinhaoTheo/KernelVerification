import sys, json
sys.path.insert(0, "/root/cases/case_149")
import numpy as np, torch, kernel

bits = np.array([i for i in range(65536) if (i >> 10) != 0x1F], dtype=np.uint16)
x_np = np.frombuffer(bits.tobytes(), dtype=np.float16).copy()
x = torch.from_numpy(x_np).to("cuda")
diag = {"numel": int(x.numel()), "isfinite": bool(torch.isfinite(x).all().item())}
if not diag["isfinite"]:
    bad = (~torch.isfinite(x)).nonzero().flatten()
    diag["n_nonfinite"] = int(bad.numel())
    diag["first_nonfinite_bits"] = [hex(int(bits[i])) for i in bad[:5].tolist()]
    print(json.dumps(diag))
else:
    out = kernel.run(x)
    ref = kernel.reference(x)
    ok = out.view(torch.int16) == ref.view(torch.int16)
    mismatches = (~ok).nonzero().flatten()
    n_mismatch = int(mismatches.numel())
    fail = []
    for idx in mismatches[:10].tolist():
        fail.append({"x": x[idx].item(),
                     "x_bits": hex(int(bits[idx])),
                     "got_bits": hex(int(out[idx].view(torch.int16).item()) & 0xFFFF),
                     "got": out[idx].item(),
                     "want_bits": hex(int(ref[idx].view(torch.int16).item()) & 0xFFFF),
                     "want": ref[idx].item()})
    print(json.dumps({"n_tested": int(x.numel()), "n_mismatch": n_mismatch,
                      "first_failures": fail}))
