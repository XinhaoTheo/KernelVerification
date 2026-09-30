import sys, json, importlib.util, traceback
import numpy as np
import torch

result = {"device": None}
try:
    spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_a/kernel.py")
    kern = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kern)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    result["device"] = dev
    if dev == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(0)

    x, a, b = kern.make_inputs(dev)
    result["shapes"] = [list(x.shape), list(a.shape), list(b.shape)]
    result["dtypes"] = [str(x.dtype), str(a.dtype), str(b.dtype)]
    result["contiguous"] = [x.is_contiguous(), a.is_contiguous(), b.is_contiguous()]
    result["x_norm"] = float(x.double().norm().item())

    # Candidate kernel on the real GPU
    out = kern.run(x, a, b)
    if dev == "cuda":
        torch.cuda.synchronize()
    out_np = out.double().cpu().numpy()

    # Reference per contract: A@x + B@x on original float32 inputs, evaluated in float64
    ref_a = a.double() @ x.double()
    ref_b = b.double() @ x.double()
    ref = ref_a + ref_b
    ref_np = ref.cpu().numpy()

    den = max(float(np.linalg.norm(ref_np)), 0.001 * np.sqrt(64))
    E = float(np.linalg.norm(out_np - ref_np)) / den
    result["E_metric"] = E
    result["ref_norm"] = float(np.linalg.norm(ref_np))
    result["denominator"] = den
    result["all_finite"] = bool(np.isfinite(out_np).all())
    result["max_abs_err"] = float(np.max(np.abs(out_np - ref_np)))
    result["ref_entry_mean_abs"] = float(np.mean(np.abs(ref_np)))

    # Float64 simulation of the kernel's quantization path (isolates quantization error
    # from Triton arithmetic): scale = max|row|/7, round-half-up, clamp to [-7,7]
    a64 = a.double().cpu().numpy(); b64 = b.double().cpu().numpy(); x64 = x.double().cpu().numpy()
    sa = np.abs(a64).max(axis=1, keepdims=True) / 7.0
    sb = np.abs(b64).max(axis=1, keepdims=True) / 7.0
    qa = np.clip(np.floor(a64 / sa + 0.5), -7.0, 7.0)
    qb = np.clip(np.floor(b64 / sb + 0.5), -7.0, 7.0)
    sim_a = ((qa * sa) * x64[None, :]).sum(axis=1)
    sim_b = ((qb * sb) * x64[None, :]).sum(axis=1)
    sim = sim_a + sim_b
    result["E_simulated_quantization"] = float(np.linalg.norm(sim - ref_np)) / den
    result["kernel_vs_sim_max_abs"] = float(np.max(np.abs(out_np - sim)))
    result["kernel_vs_sim_norm"] = float(np.linalg.norm(out_np - sim))

    # Per-branch relative errors of the quantized computation (context for the contract's
    # remark that either branch may individually exceed 0.1)
    ra_np = ref_a.cpu().numpy(); rb_np = ref_b.cpu().numpy()
    result["branchA_ref_norm"] = float(np.linalg.norm(ra_np))
    result["branchB_ref_norm"] = float(np.linalg.norm(rb_np))
    result["branchA_rel_err_quant"] = float(np.linalg.norm(sim_a - ra_np) / max(np.linalg.norm(ra_np), 1e-12))
    result["branchB_rel_err_quant"] = float(np.linalg.norm(sim_b - rb_np) / max(np.linalg.norm(rb_np), 1e-12))

    # Quantization context stats
    result["mean_rowmax_abs_A"] = float(np.abs(a64).max(axis=1).mean())
    result["mean_rowmax_abs_B"] = float(np.abs(b64).max(axis=1).mean())
    result["mean_step_A"] = float((np.abs(a64).max(axis=1) / 7.0).mean())
    result["mean_step_B"] = float((np.abs(b64).max(axis=1) / 7.0).mean())

    # Determinism check: run again
    out2 = kern.run(x, a, b)
    if dev == "cuda":
        torch.cuda.synchronize()
    result["deterministic"] = bool(np.array_equal(out2.double().cpu().numpy(), out_np))

    result["contract_E_le_0.1"] = bool(E <= 0.1)
    result["verdict_signal"] = "VIOLATION" if E > 0.1 else "COMPLIANT"
except Exception as e:
    result["error"] = traceback.format_exc()

print(json.dumps(result, indent=2))
