
import torch, json, math, sys
sys.path.insert(0, "/root/cases/case_114")
import kernel as K

dev = "cuda"
# Worst-case precision: Q=1, lengths=512, window=0 (full causal, 512 allowed), D=128, GROUP=1
B, Q, HQ, HK, D = 1, 1, 1, 1, 128
S, P = 16, 32  # 32 pages * 16 = 512 slots, exactly covers positions 0..511
window = 0
length = 512
g = torch.Generator(device=dev).manual_seed(7)
q = (torch.rand((B, Q, HQ, D), device=dev, generator=g) * 0.02 - 0.01).half()  # tiny q -> near-uniform softmax
k = (torch.rand((P, S, HK, D), device=dev, generator=g) * 0.02 - 0.01).half()
v = (torch.rand((P, S, HK, D), device=dev, generator=g) * 2 - 1).half()      # random v, targets moderate
# extra: adversarial cancelling v -> targets near zero, amplifies FP16 weight rounding
v_c = (torch.rand((P, S, HK, D), device=dev, generator=g) * 0.1).half()
v_c[:, :, :, 0::2] *= -1.0
v_c = v_c * (torch.arange(512, device=dev).view(1, 32, 16, 1) % 2 * 2 - 1).half()  # alternate sign by position

table = torch.arange(0, 32, device=dev, dtype=torch.int32).unsqueeze(0)
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
