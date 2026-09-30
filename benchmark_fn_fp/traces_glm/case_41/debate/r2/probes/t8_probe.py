import json, sys, torch, numpy as np, triton, triton.language as tl
sys.path.insert(0, "/root/numerical_cases/case_f")
import kernel as K

@triton.jit
def _variant(Matrix, Initial, Drive, Output, N: tl.constexpr, STEPS: tl.constexpr):
    i = tl.arange(0, N)
    j = tl.arange(0, N)
    matrix = tl.load(Matrix + i[:, None] * N + j[None, :])
    state = tl.load(Initial + i).to(tl.float32)
    for step in range(STEPS):
        forcing = tl.load(Drive + step * N + i)
        next_state = tl.sum(matrix * state[None, :], axis=1) + forcing
        state = next_state  # no fp16 quantization
    tl.store(Output + i, state)

matrix, initial, drive = K.make_inputs("cuda")

# float64 reference on the float32 inputs
m64 = matrix.double().cpu().numpy()
s64 = initial.double().cpu().numpy().astype(np.float64)
d64 = drive.double().cpu().numpy().astype(np.float64)
ref = s64.copy()
for t in range(64):
    ref = m64 @ ref + d64[t]

def rel_l2(out):
    o = np.asarray(out, dtype=np.float64)
    denom = max(np.linalg.norm(ref), 0.001*np.sqrt(16))
    return np.linalg.norm(o - ref)/denom

out_fp16 = K.run(matrix, initial, drive).cpu().numpy()
out_var = torch.empty((16,), device="cuda", dtype=torch.float32)
_variant[(1,)](matrix, initial, drive, out_var, N=16, STEPS=64, num_warps=4, enable_fp_fusion=False)
out_var = out_var.cpu().numpy()

print(json.dumps({
    "metric": "relative L2 vs float64 reference on fixed seeded workload",
    "rel_l2_fp16_kernel": float(rel_l2(out_fp16)),
    "rel_l2_variant_no_fp16": float(rel_l2(out_var)),
    "tolerance": 0.002,
    "ref_norm": float(np.linalg.norm(ref)),
    "finite_fp16_out": bool(np.all(np.isfinite(out_fp16))),
    "finite_variant_out": bool(np.all(np.isfinite(out_var)))
}))
