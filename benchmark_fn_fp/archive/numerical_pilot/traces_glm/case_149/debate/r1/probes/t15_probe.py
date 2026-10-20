import torch, numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/pilot_cases/case_86/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x, w = k.make_inputs("cuda")
wf = w.double().cpu().numpy()
# float64 generator quantizer
s64 = np.max(np.abs(wf), axis=1, keepdims=True) / 7.0
q64 = np.clip(np.floor(wf/s64 + 0.5), -7, 7) * s64
# float32 quantizer emulating kernel (scale then divide in fp32)
w32 = w.cpu().numpy()
s32 = (np.max(np.abs(w32), axis=1, keepdims=True) / 7.0).astype(np.float32)
q32 = (np.clip(np.floor((w32/s32).astype(np.float32) + 0.5), -7, 7) * s32).astype(np.float32)
lvl64 = np.rint(q64/s64); lvl32 = np.rint(q32.astype(np.float64)/s32)
mismatch = int((lvl64 != lvl32).sum())
maxelem_lvl = [lvl32[i][np.argmax(np.abs(w32[i]))] for i in range(w32.shape[0])]
extra_err = float(np.abs(q32.astype(np.float64) - wf).max())
bound_err = float((s64/2).max())
print(json.dumps({"level_mismatch_count": mismatch, "total_elements": int(wf.size), "max_elem_levels_distinct_from_7": int(sum(1 for l in maxelem_lvl if abs(l-7.0)>1e-6)), "quant_err_max_abs": extra_err, "scale_half_bound": bound_err, "exceeds_bound": extra_err > bound_err}))