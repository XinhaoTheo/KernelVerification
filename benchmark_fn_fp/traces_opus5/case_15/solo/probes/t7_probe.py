
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_15/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref(x, gs):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    for s in range(0, n_cols, gs):
        e = min(s+gs, n_cols)
        blk = x[:, s:e]
        absmax = blk.abs().amax(dim=1, keepdim=True)
        scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax/127.0)
        q = torch.round(blk/scale).clamp(-127,127)
        y[:, s:e] = q*scale
    return y

res = {}
for n_cols in [100, 128, 64, 65, 200]:
    torch.manual_seed(0)
    x = torch.randn(4, n_cols, device='cuda')
    out = m.group_quant_dequant(x, 64)
    r = ref(x, 64)
    tail_start = (n_cols//64)*64
    d = (out-r).abs()
    res[n_cols] = {
        "max_abs_err": float(d.max()),
        "tail_cols": n_cols - tail_start,
        "tail_out_all_zero": bool(out[:, tail_start:].eq(0).all()) if tail_start < n_cols else None,
        "tail_ref_absmax": float(r[:, tail_start:].abs().max()) if tail_start < n_cols else None,
        "head_max_abs_err": float(d[:, :tail_start].max()) if tail_start > 0 else None,
    }
print(json.dumps(res, indent=1))
