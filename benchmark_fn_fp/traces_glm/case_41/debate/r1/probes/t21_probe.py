import torch, json, math, sys
sys.path.insert(0, "/root/cases/case_114")
import kernel as K

dev = "cuda"
B, Q, HQ, HK, D = 1, 1, 1, 1, 128
S, P = 16, 32
window = 0
length = 512
gg = torch.Generator().manual_seed(7)
q = ((torch.rand((B, Q, HQ, D), generator=gg) * 0.02 - 0.01).to(dev)).half()
k = ((torch.rand((P, S, HK, D), generator=gg) * 0.02 - 0.01).to(dev)).half()
v = ((torch.rand((P, S, HK, D), generator=gg) * 2 - 1).to(dev)).half()
# adversarial cancelling v: alternate sign per logical position (identity table: t = page*S + slot)
sign = ((torch.arange(P * S) % 2 * 2 - 1).float()).view(P, S, 1, 1).to(dev).half()
v_c = ((torch.rand((P, S, HK, D), generator=gg) * 0.2).to(dev)).half() * sign

table = torch.arange(0, P, dtype=torch.int32).unsqueeze(0).to(dev)
lengths = torch.tensor([length], dtype=torch.int32, device=dev)

results = {}
for name, vt in (("random_v", v), ("cancelling_v", v_c)):
    K.validate_inputs(q, k, vt, table, lengths, window)
    out = K.run(q, k, vt, table, lengths, window)
    torch.cuda.synchronize()
    ref = K.reference(q, k, vt, table, lengths, window)
    abs_err = (out.double() - ref).abs()
    tol = 0.003 + 0.003 * ref.abs()
    results[name] = {
        "max_abs_err": float(abs_err.max()),
        "tolerance_violation_count": int((abs_err > tol).sum()),
        "target_abs_max": float(ref.abs().max()),
        "target_abs_min": float(ref.abs().min()),
        "error_ratio_max": float((abs_err / tol).max()),
    }
print(json.dumps({"claim": "c2", "window": 0, "length": length, "Q": Q, "D": D,
                  "n_allowed_positions": 512, "results": results}))