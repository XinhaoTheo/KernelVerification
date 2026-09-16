
import importlib.util, json, numpy as np, torch

spec = importlib.util.spec_from_file_location("kk", "/root/cases/case_33/kernel.py")
kk = importlib.util.module_from_spec(spec); spec.loader.exec_module(kk)

dev = 'cuda'
rng = np.random.default_rng(0)
M, N, K = 64, 64, 48
gs = 32
G = -(-K // gs)   # ceil = 2 contract group rows

q = rng.integers(0, 16, size=(K, N)).astype(np.uint32)
rows = -(-K // 8)
pk = np.zeros((rows, N), dtype=np.uint32)
for k in range(K):
    pk[k // 8] |= (q[k] & 15) << (4 * (k % 8))
b = torch.from_numpy(np.ascontiguousarray(pk.view(np.int32))).to(dev)

# zeros packed 4-bit along N (kernel's own convention). grossly distinct rows.
zq = np.zeros((G, N), dtype=np.uint32); zq[0] = 2; zq[1] = 13
zp = np.zeros((G, N // 8), dtype=np.uint32)
for n in range(N):
    zp[:, n // 8] |= (zq[:, n] & 15) << (4 * (n % 8))

sc = np.zeros((G, N), dtype=np.float32); sc[0] = 1.0; sc[1] = 8.0

a = torch.randn(M, K, device=dev, dtype=torch.float32)

# reference: kernel's dequant conventions (packed zeros, zero = nibble+1) but CONTRACT g_idx
gidx = np.arange(K) // gs
deq = (q.astype(np.float64) - (zq[gidx].astype(np.float64) + 1.0)) * sc[gidx].astype(np.float64)
ref = a.double().cpu().numpy() @ deq

# run 1: ragged path (num_groups = 48//32 = 1, clamp max = 0)
c_bug = kk.gptq_matmul(a, b,
                       torch.from_numpy(sc).to(dev),
                       torch.from_numpy(np.ascontiguousarray(zp.view(np.int32))).to(dev),
                       gs, 4)
torch.cuda.synchronize()

# run 2: mathematically identical grouping expressed with gs=16 and rows [p0,p0,p1]
sc3 = np.ascontiguousarray(np.stack([sc[0], sc[0], sc[1]]))
zp3 = np.ascontiguousarray(np.stack([zp[0], zp[0], zp[1]]))
c_eq = kk.gptq_matmul(a, b,
                      torch.from_numpy(sc3).to(dev),
                      torch.from_numpy(np.ascontiguousarray(zp3.view(np.int32))).to(dev),
                      16, 4)
torch.cuda.synchronize()

cb = c_bug.double().cpu().numpy(); ce = c_eq.double().cpu().numpy()
scale_ref = float(np.abs(ref).max())
out = {
    "shape": [M, N, K], "group_size": gs, "contract_group_rows": G,
    "wrapper_num_groups": K // gs,
    "g_idx_max_wrapper": int(min(K // gs - 1, (K - 1) // gs)),
    "ref_max_abs": scale_ref,
    "ragged_max_abs_err_vs_ref": float(np.abs(cb - ref).max()),
    "ragged_rel_err_vs_ref": float(np.abs(cb - ref).max() / scale_ref),
    "nonragged_equiv_max_abs_err_vs_ref": float(np.abs(ce - ref).max()),
    "nonragged_equiv_rel_err_vs_ref": float(np.abs(ce - ref).max() / scale_ref),
    "ragged_vs_equiv_max_abs_diff": float(np.abs(cb - ce).max()),
    "frac_entries_err_gt_1pct": float((np.abs(cb - ref) > 0.01 * scale_ref).mean()),
    "metric_reason": "reference emulates the kernel's own dequant conventions (zeros packed 4-bit along N, zero=nibble+1) so only the group-index assignment differs; the gs=16 duplicated-row call is the confound-free self-differential control",
}
print(json.dumps(out))
