import torch, json, numpy as np, sys
sys.path.insert(0, "/root/pilot_cases/case_98")
import kernel as K

a, b = K.make_inputs("cuda")
out = K.run(a, b).float().cpu().numpy()
a32 = a.float().cpu().numpy(); b32 = b.float().cpu().numpy()

# fp64 reference
ref = np.zeros((640,32)); h = np.zeros(32)
for t in range(640):
    h = a32[t].astype(np.float64) * h + b32[t].astype(np.float64)
    ref[t] = h

# CPU simulation of quantized recurrence (fp32 compute, fp16 round via torch)
sim = np.zeros((640,32)); hs = np.zeros(32, dtype=np.float32)
for t in range(640):
    hs = (a32[t].astype(np.float32) * hs + b32[t]).astype(np.float32)
    hs = torch.from_numpy(hs).to(torch.float16).to(torch.float32).numpy()
    sim[t] = hs

per_t = np.linalg.norm(out - ref, axis=1)
err_full = np.linalg.norm((out-ref).ravel())
err_late = np.linalg.norm((out-ref)[400:])
err_early = np.linalg.norm((out-ref)[:400])
bitwise_match = bool((out == sim).all())
res = {
    "metric": "per-timestep ||out[t]-ref[t]|| vs late-time fraction; bitwise CPU fp16-round sim match",
    "err_norm_early_t0_400": float(err_early),
    "err_norm_late_t400_640": float(err_late),
    "late_fraction": float(err_late/err_full),
    "per_t_err_first": float(per_t[0]), "per_t_err_mid": float(per_t[320]), "per_t_err_last": float(per_t[639]),
    "per_t_err_max": float(per_t.max()), "per_t_err_max_t": int(per_t.argmax()),
    "mean_sign_out_last100": float(np.sign(out[-100:]-ref[-100:]).mean()),
    "bitwise_match_to_cpu_sim": bitwise_match,
    "late_states_dominate": bool(err_late > err_early),
}
print(json.dumps(res))