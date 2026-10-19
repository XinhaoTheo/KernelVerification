import json, os, sys, glob, importlib.util, traceback
import torch

RESULT = {
    "claim_id": "c3",
    "metric": "distinct int32 bit patterns of blocked_sum(x)[0] across repeated calls on the same off-grid torch.randn tensor (N=65536, default block=1024); c3 confirmed iff at least 2 distinct patterns are observed",
    "note": "off-grid inputs are outside the problem.txt domain; this probe tests the claim's conditional mechanism (atomic-order rounding drift) only, as a generalization note",
}

def find_kernel_path():
    for c in ["/root/cases/case_35/kernel.py",
              os.path.join(os.getcwd(), "kernel.py"),
              "kernel.py",
              os.path.join(os.getcwd(), "case_35", "kernel.py")]:
        if os.path.isfile(c):
            return c
    for h in sorted(glob.glob("/root/cases/*/kernel.py")):
        if "case_35" in h:
            return h
    return None

try:
    p = find_kernel_path()
    if p is None:
        raise RuntimeError("kernel.py not found in probe environment")
    spec = importlib.util.spec_from_file_location("case35_kernel_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    blocked_sum = mod.blocked_sum
    RESULT["kernel_path"] = p
    import triton
    RESULT["triton_version"] = triton.__version__
    RESULT["torch_version"] = torch.__version__
    if not torch.cuda.is_available():
        RESULT["gpu_available"] = False
        RESULT["error"] = "no CUDA device available; Triton kernel cannot be executed"
        print(json.dumps(RESULT))
        sys.exit(0)
    RESULT["gpu_available"] = True
    RESULT["gpu_name"] = torch.cuda.get_device_name(0)
    RESULT["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    dev = torch.device("cuda")
    N = 65536
    REPS = 250
    tensors = {}
    for seed in (20260921, 42):
        g = torch.Generator().manual_seed(seed)
        tensors["randn_seed%d" % seed] = torch.randn(N, generator=g, dtype=torch.float32).to(dev)
    per_tensor = {}
    any_drift = False
    for name, x in tensors.items():
        offgrid = bool(((x.to(torch.float64) * 128.0) % 1.0 != 0).any().item())
        pats = []
        vals = []
        for _ in range(REPS):
            o = blocked_sum(x)
            vals.append(float(o.item()))
            pats.append(int(o.view(torch.int32).item()))
        d = sorted(set(pats))
        dvals = sorted(set(vals))
        ref64 = float(x.to(torch.float64).sum().item())
        max_abs_dev = max(abs(v - ref64) for v in dvals) if dvals else None
        same_sign = (d[0] < 0) == (d[-1] < 0)
        if len(d) > 1 and same_sign:
            spread = d[-1] - d[0]
        elif len(d) == 1:
            spread = 0
        else:
            spread = None
        per_tensor[name] = {
            "off_grid_verified": offgrid,
            "reps": REPS,
            "n_distinct_bit_patterns": len(d),
            "distinct_patterns_hex": [hex(v) for v in d[:10]],
            "distinct_values": dvals[:10],
            "float64_reference_sum": ref64,
            "max_abs_dev_from_ref64": max_abs_dev,
            "bit_pattern_spread_units": spread,
        }
        if len(d) > 1:
            any_drift = True
    RESULT["per_tensor"] = per_tensor
    RESULT["total_kernel_calls"] = REPS * len(tensors)
    RESULT["any_run_to_run_bit_drift_offgrid"] = any_drift
    RESULT["observed_signal"] = ("off-grid repeated calls produced at least 2 distinct bit patterns"
                                 if any_drift else
                                 "off-grid repeated calls all produced identical bits within tested reps")
except Exception as e:
    RESULT["error"] = type(e).__name__ + ": " + str(e)
    RESULT["traceback"] = traceback.format_exc()[-1500:]
print(json.dumps(RESULT))