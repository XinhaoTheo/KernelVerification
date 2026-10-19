
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_07/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
K, N, group = 128, 64, 32
G = K // group
dev = 'cuda'

q  = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)
zs = torch.randint(0, 16, (G, N), device=dev, dtype=torch.int32)   # stored zero code
scales = (torch.rand((G, N), device=dev, dtype=torch.float16) * 0.1 + 0.01)
g_idx = (torch.arange(K, device=dev) // group).to(torch.int32)

# pack weights along K: packed[i, n] holds q[i*8+j, n] at shift 4*j
qw = torch.zeros((K // 8, N), device=dev, dtype=torch.int32)
qr = q.reshape(K // 8, 8, N)
for j in range(8):
    qw |= (qr[:, j, :] & 0xF) << (4 * j)

# pack zeros along N: qz[g, n//8] holds zs[g, n] at shift 4*(n%8)
qz = torch.zeros((G, N // 8), device=dev, dtype=torch.int32)
zr = zs.reshape(G, N // 8, 8)
for j in range(8):
    qz |= (zr[:, :, j] & 0xF) << (4 * j)

# references
z_plus1 = (zs + 1)[g_idx.long()]
z_raw   = zs[g_idx.long()]
sc      = scales[g_idx.long()].float()
deq_p1  = (q.float() - z_plus1.float()) * sc
deq_raw = (q.float() - z_raw.float()) * sc

out = {}
for name, adt in [("fp16", torch.float16), ("fp32", torch.float32)]:
    a = torch.eye(K, device=dev, dtype=adt)
    try:
        c = m.gptq_matmul(a, qw, scales, qz, g_idx, bits=4).float()
    except Exception as e:
        out[name] = {"error": repr(e)[:300]}
        continue
    e1 = (c - deq_p1).abs()
    e2 = (c - deq_raw).abs()
    den = deq_p1.abs().clamp_min(1e-6)
    out[name] = {
        "max_abs_err_vs_zero_plus1": float(e1.max()),
        "max_rel_err_vs_zero_plus1": float((e1 / den).max()),
        "max_abs_err_vs_zero_raw": float(e2.max()),
        "frac_elems_err_gt_1e-3_p1": float((e1 > 1e-3).float().mean()),
        "deq_absmax": float(deq_p1.abs().max()),
    }
    # per-row / per-col error structure (helps spot axis bugs)
    rowerr = e1.max(dim=1).values
    colerr = e1.max(dim=0).values
    out[name]["rows_bad"] = [int(i) for i in torch.nonzero(rowerr > 1e-3).flatten()[:20]]
    out[name]["cols_bad"] = [int(i) for i in torch.nonzero(colerr > 1e-3).flatten()[:20]]

print(json.dumps(out, indent=2))
