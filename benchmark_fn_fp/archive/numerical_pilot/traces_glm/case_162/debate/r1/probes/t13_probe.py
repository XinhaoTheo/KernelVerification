import sys, json, numpy as np, torch
sys.path.insert(0, "/root/pilot_cases/case_93")
import kernel as K

x, w = K.make_inputs("cuda")
out = K.run(x, w)
torch.cuda.synchronize()
triton = out.double().cpu().numpy()
wf = w.double().cpu().numpy()
xf = x.double().cpu().numpy()
ref = wf @ xf

# float64 emulation of the kernel's quantization
scale64 = np.abs(wf).max(axis=1, keepdims=True) / 7.0
qi64 = np.clip(np.floor(wf / scale64 + 0.5), -7, 7)
q64 = qi64 * scale64
emul64 = q64 @ xf

# float32 emulation (same ops in float32, as Triton does)
w32 = w.cpu().numpy().astype(np.float32)
x32 = x.cpu().numpy().astype(np.float32)
scale32 = (np.abs(w32).max(axis=1, keepdims=True) / np.float32(7.0)).astype(np.float32)
qi32 = np.clip(np.floor(w32 / scale32 + np.float32(0.5)), -7, 7)
q32 = (qi32 * scale32).astype(np.float32)
emul32 = q32.astype(np.float64) @ xf

code_diff = int((qi32 != qi64).sum())
den = max(np.linalg.norm(ref), 0.001*np.sqrt(ref.size))
E_triton = np.linalg.norm(triton - ref) / den
E_emul64 = np.linalg.norm(emul64 - ref) / den
E_emul32 = np.linalg.norm(emul32 - ref) / den
print(json.dumps({
    "metric": "quantization_code_comparison_and_E_shift",
    "qi_code_differences_float32_vs_float64": code_diff,
    "total_codes": int(qi64.size),
    "E_triton_actual": float(E_triton),
    "E_emulated_float64": float(E_emul64),
    "E_emulated_float32": float(E_emul32),
    "max_abs_out_diff_triton_vs_emul64": float(np.abs(triton - emul64).max()),
    "budget": 0.12
}))