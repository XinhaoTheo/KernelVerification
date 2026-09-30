import json, math, importlib.util, traceback
import numpy as np
import torch

result = {"claim": "c2", "probe": "cancellation mechanism: PERMUTATION pairing, independent NumPy regeneration, permutation controls"}

def q_emulate(M):
    # fp32 emulation of kernel.py lines 12-15 (per-row 15-level +-7 quantization, round-half-up)
    Mf = np.ascontiguousarray(M, dtype=np.float32)
    s = (np.abs(Mf).max(axis=1, keepdims=True) / np.float32(7.0)).astype(np.float32)
    q = np.floor(Mf / s + np.float32(0.5))
    q = np.clip(q, np.float32(-7.0), np.float32(7.0))
    return (q * s).astype(np.float32)

def branch_errs(A, B, x):
    Aq = q_emulate(A)
    Bq = q_emulate(B)
    xd = x.astype(np.float64)
    refA = A.astype(np.float64) @ xd
    refB = B.astype(np.float64) @ xd
    outA = Aq.astype(np.float64) @ xd
    outB = Bq.astype(np.float64) @ xd
    return outA - refA, outB - refB, refA + refB, outA + outB

def E_of(errA, errB, ref):
    comb = float(np.linalg.norm(errA + errB))
    denom = max(float(np.linalg.norm(ref)), 0.001 * math.sqrt(64))
    return comb / denom, comb, denom

try:
    spec = importlib.util.spec_from_file_location("case_b_kernel", "/root/cases/case_b/kernel.py")
    kmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kmod)
    PERM = list(kmod.PERMUTATION)
    result["numpy_version"] = np.__version__
    result["permutation_is_bijection"] = bool(sorted(PERM) == list(range(64)))

    # independent NumPy re-derivation of the generator (kernel.py lines 28-42)
    rng = np.random.Generator(np.random.PCG64(921000))
    x64 = rng.standard_normal(128)
    x64 /= np.linalg.norm(x64)
    mats = []
    for _ in range(2):
        w = rng.standard_normal((64, 128))
        target = 0.5 + 0.02 * rng.standard_normal(64)
        projection = np.sum(w * x64[None, :], axis=1, dtype=np.float64)
        w += ((target - projection) / np.sum(x64 * x64))[:, None] * x64[None, :]
        mats.append(w.astype(np.float32))
    x_np = x64.astype(np.float32)
    A_np = mats[0]
    Bpre_np = mats[1]
    B_np = Bpre_np[PERM].copy()

    # bitwise input identity vs make_inputs
    x_t, A_t, B_t = kmod.make_inputs(device="cpu")
    result["inputs_bitwise_equal_make_inputs"] = {
        "x": bool(np.array_equal(x_np, x_t.numpy())),
        "A": bool(np.array_equal(A_np, A_t.numpy())),
        "B": bool(np.array_equal(B_np, B_t.numpy())),
    }

    # actual workload (B rows reordered by PERMUTATION)
    errA, errB, ref, out_emu = branch_errs(A_np, B_np, x_np)
    E_act, comb_act, denom = E_of(errA, errB, ref)
    result.update({
        "E_actual_permutation": E_act,
        "combined_err_norm_actual": comb_act,
        "denominator": denom,
        "corr_actual_permutation": float(np.corrcoef(errA, errB)[0, 1]),
        "errA_norm": float(np.linalg.norm(errA)),
        "errB_norm": float(np.linalg.norm(errB)),
        "per_row_combined_err_absmax": float(np.abs(errA + errB).max()),
        "per_row_branch_err_absmean": float((np.abs(errA) + np.abs(errB)).mean()),
    })

    # identity pairing (pre-permutation B)
    errA_id, errB_id, ref_id, _ = branch_errs(A_np, Bpre_np, x_np)
    E_id, _, _ = E_of(errA_id, errB_id, ref_id)
    result["E_identity_pairing"] = E_id
    result["corr_identity_pairing"] = float(np.corrcoef(errA_id, errB_id)[0, 1])

    # 200 random pairings
    rrng = np.random.Generator(np.random.PCG64(20260923))
    Es = []
    corrs = []
    for _ in range(200):
        p = rrng.permutation(64)
        eA, eB, rf, _ = branch_errs(A_np, Bpre_np[p], x_np)
        E_p, _, _ = E_of(eA, eB, rf)
        Es.append(E_p)
        corrs.append(float(np.corrcoef(eA, eB)[0, 1]))
    Es = np.asarray(Es)
    corrs = np.asarray(corrs)
    result.update({
        "random_perm_count": 200,
        "random_perm_E_mean": float(Es.mean()),
        "random_perm_E_std": float(Es.std()),
        "random_perm_E_min": float(Es.min()),
        "random_perm_E_max": float(Es.max()),
        "random_perm_frac_E_within_bound": float((Es <= 0.1).mean()),
        "random_perm_corr_mean": float(corrs.mean()),
        "random_perm_corr_absmax": float(np.abs(corrs).max()),
    })

    # anti-sort pairing hypothesis: PERMUTATION constructed to anti-match branch errors
    eA_pre, eB_pre, _, _ = branch_errs(A_np, Bpre_np, x_np)
    idxA_asc = np.argsort(eA_pre)
    idxB_desc = np.argsort(-eB_pre)
    p_anti = np.empty(64, dtype=np.int64)
    p_anti[idxA_asc] = idxB_desc
    eA_a, eB_a, ref_a, _ = branch_errs(A_np, Bpre_np[p_anti], x_np)
    E_anti, _, _ = E_of(eA_a, eB_a, ref_a)
    result.update({
        "anti_sort_perm_E": E_anti,
        "anti_sort_perm_corr": float(np.corrcoef(eA_a, eB_a)[0, 1]),
        "anti_sort_perm_exact_match_count": int((p_anti == np.asarray(PERM)).sum()),
    })

    # GPU cross-check with the actual kernel
    gpu = torch.cuda.is_available()
    result["gpu_available"] = bool(gpu)
    if gpu:
        result["gpu_device"] = torch.cuda.get_device_name(0)
        xg, Ag, Bg = kmod.make_inputs(device="cuda")
        out = kmod.run(xg, Ag, Bg)
        torch.cuda.synchronize()
        out_np = out.double().cpu().numpy()
        result["kernel_ran"] = True
        result["kernel_E_crosscheck"] = float(np.linalg.norm(out_np - ref)) / denom
        result["kernel_vs_numpy_emu_max_absdiff"] = float(np.abs(out_np - out_emu).max())
    else:
        result["kernel_ran"] = False
except Exception as e:
    result["probe_error"] = f"{type(e).__name__}: {e}"
    result["traceback"] = traceback.format_exc()[-2000:]

print(json.dumps(result))