import json, math, importlib.util, traceback
import numpy as np
import torch

result = {"claim": "c1", "probe": "exact contract metric E on fixed seed-921000 workload"}
try:
    spec = importlib.util.spec_from_file_location("case_b_kernel", "/root/cases/case_b/kernel.py")
    kmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kmod)
    result["torch_version"] = torch.__version__
    try:
        import triton
        result["triton_version"] = triton.__version__
    except Exception as e:
        result["triton_version"] = f"import failed: {e}"

    gpu = torch.cuda.is_available()
    result["gpu_available"] = bool(gpu)
    device = "cuda" if gpu else "cpu"
    if gpu:
        result["gpu_device"] = torch.cuda.get_device_name(0)

    x, A, B = kmod.make_inputs(device=device)
    result["shapes"] = [list(t.shape) for t in (x, A, B)]
    result["dtypes"] = [str(t.dtype) for t in (x, A, B)]
    result["contiguous"] = [bool(t.is_contiguous()) for t in (x, A, B)]
    result["x_l2_fp64"] = float(np.linalg.norm(x.cpu().numpy().astype(np.float64)))

    out = None
    if gpu:
        try:
            out = kmod.run(x, A, B)
            torch.cuda.synchronize()
            result["kernel_ran"] = True
        except Exception as e:
            result["kernel_ran"] = False
            result["kernel_error"] = f"{type(e).__name__}: {e}"
    else:
        result["kernel_ran"] = False
        result["kernel_error"] = "cuda unavailable; kernel not executed"

    # fp64 reference on the original float32 inputs
    xd, Ad, Bd = x.double(), A.double(), B.double()
    ref = Ad @ xd + Bd @ xd
    refA = Ad @ xd
    refB = Bd @ xd
    ref_norm = float(ref.norm())
    denom = max(ref_norm, 0.001 * math.sqrt(64))

    # fp32 emulation of the kernel's per-row quantization (kernel.py lines 12-15)
    sa = A.abs().amax(dim=1, keepdim=True) / 7.0
    sb = B.abs().amax(dim=1, keepdim=True) / 7.0
    qa = torch.clamp(torch.floor(A / sa + 0.5), -7.0, 7.0)
    qb = torch.clamp(torch.floor(B / sb + 0.5), -7.0, 7.0)
    Aq, Bq = qa * sa, qb * sb
    out_emu = (Aq * x[None, :]).sum(dim=1) + (Bq * x[None, :]).sum(dim=1)

    # quantization statistics
    ea = Aq.double() - Ad
    eb = Bq.double() - Bd
    pca = torch.floor(A / sa + 0.5)
    pcb = torch.floor(B / sb + 0.5)

    result.update({
        "ref_norm": ref_norm,
        "denominator": denom,
        "refA_norm": float(refA.norm()),
        "refB_norm": float(refB.norm()),
        "ref_row_mean": float(ref.mean()),
        "ref_row_std": float(ref.std()),
        "ref_row_absmax": float(ref.abs().max()),
        "sa_mean": float(sa.mean()), "sa_min": float(sa.min()), "sa_max": float(sa.max()),
        "sb_mean": float(sb.mean()), "sb_min": float(sb.min()), "sb_max": float(sb.max()),
        "qerr_a_min": float(ea.min()), "qerr_a_max": float(ea.max()), "qerr_a_std": float(ea.std()),
        "qerr_b_min": float(eb.min()), "qerr_b_max": float(eb.max()), "qerr_b_std": float(eb.std()),
        "clamp_active_count_a": int(((pca > 7.0) | (pca < -7.0)).sum()),
        "clamp_active_count_b": int(((pcb > 7.0) | (pcb < -7.0)).sum()),
        "emu_errA_norm": float((Aq.double() @ xd - refA).norm()),
        "emu_errB_norm": float((Bq.double() @ xd - refB).norm()),
    })

    if out is not None:
        err = out.double() - ref
        err_norm = float(err.norm())
        E = err_norm / denom
        result.update({
            "output_finite": bool(torch.isfinite(out).all()),
            "output_shape": list(out.shape),
            "err_norm": err_norm,
            "E": E,
            "contract_bound": 0.1,
            "E_within_bound": bool(E <= 0.1),
            "max_abs_row_err": float(err.abs().max()),
            "emu_vs_kernel_max_absdiff": float((out_emu.double() - out.double()).abs().max()),
        })
    else:
        result["emu_only_E"] = float((out_emu.double() - ref).norm() / denom)
except Exception as e:
    result["probe_error"] = f"{type(e).__name__}: {e}"
    result["traceback"] = traceback.format_exc()[-2000:]

print(json.dumps(result))