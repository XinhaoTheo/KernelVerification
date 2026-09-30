import importlib.util, os, sys, json, traceback, glob

def find_kernel():
    cands = [
        "/root/cases/case_a/kernel.py",
        os.path.join(os.getcwd(), "kernel.py"),
        os.path.join(os.getcwd(), "case_a", "kernel.py"),
        os.path.join(os.getcwd(), "..", "kernel.py"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    hits = sorted(glob.glob("/root/cases/*/kernel.py"))
    if hits:
        return hits[0]
    raise FileNotFoundError("kernel.py not found; tried %r" % (cands,))

def main():
    kpath = find_kernel()
    spec = importlib.util.spec_from_file_location("kernel_mod_c1", kpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import numpy as np
    import torch

    cuda_ok = bool(torch.cuda.is_available())
    device = "cuda" if cuda_ok else "cpu"
    x, a, b = mod.make_inputs(device)
    out = mod.run(x, a, b)
    if cuda_ok:
        torch.cuda.synchronize()

    xd = x.detach().cpu().numpy().astype(np.float64)
    ad = a.detach().cpu().numpy().astype(np.float64)
    bd = b.detach().cpu().numpy().astype(np.float64)

    # Reference exactly per problem.txt: y = A @ x + B @ x on the original
    # float32 inputs, evaluated in float64.
    ref = ad @ xd + bd @ xd
    refn = float(np.linalg.norm(ref))
    floor_den = 0.001 * (64 ** 0.5)
    den = max(refn, floor_den)

    outd = out.detach().cpu().numpy().astype(np.float64)
    diff = outd - ref
    num = float(np.linalg.norm(diff))
    E = num / den

    sa = np.max(np.abs(ad), axis=1) / 7.0
    sb = np.max(np.abs(bd), axis=1) / 7.0

    res = {
        "probe": "c1_runtime_E_of_run_make_inputs",
        "metric": "E = ||out - ref||_2 / max(||ref||_2, 0.001*sqrt(64)); ref = float64(A@x + B@x) on original float32 inputs",
        "metric_reason": "problem.txt judges the final combined output of run(*make_inputs()) on exactly this fixed workload: all outputs finite and E <= 0.1",
        "kernel_path": kpath,
        "seed": int(mod.SEED),
        "device": str(x.device),
        "gpu_name": (torch.cuda.get_device_name(0) if cuda_ok else None),
        "shapes": {"x": list(xd.shape), "A": list(ad.shape), "B": list(bd.shape), "out": list(outd.shape)},
        "out_dtype": str(out.dtype),
        "E_kernel": E,
        "contract_threshold": 0.1,
        "contract_pass": bool(E <= 0.1 and np.all(np.isfinite(outd))),
        "err_l2": num,
        "ref_l2": refn,
        "denominator": den,
        "denominator_floor": floor_den,
        "rms_row_err": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_row_err": float(np.max(np.abs(diff))),
        "all_outputs_finite": bool(np.all(np.isfinite(outd))),
        "ref_row_mean": float(np.mean(ref)),
        "ref_row_std": float(np.std(ref)),
        "scale_a_min_mean_max": [float(np.min(sa)), float(np.mean(sa)), float(np.max(sa))],
        "scale_b_min_mean_max": [float(np.min(sb)), float(np.mean(sb)), float(np.max(sb))],
        "x_norm": float(np.linalg.norm(xd)),
    }
    print(json.dumps(res))

try:
    main()
except Exception as e:
    print(json.dumps({
        "probe": "c1_runtime_E_of_run_make_inputs",
        "error": repr(e),
        "traceback": traceback.format_exc(),
    }))