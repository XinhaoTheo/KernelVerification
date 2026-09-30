import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_15/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

torch.manual_seed(0)
n_rows, n_cols, gs = 3, 100, 64  # not multiple of 64
x = torch.randn(n_rows, n_cols, device="cuda")
y = mod.group_quant_dequant(x, group_size=gs)

def ref(x, gs):
    outs = []
    for r in range(x.shape[0]):
        row = []
        for s in range(0, x.shape[1], gs):
            g = x[r, s:s+gs]
            m = g.abs().max()
            scale = 1.0 if m == 0 else m/127.0
            q = torch.clamp(torch.round(g/scale), -127, 127)
            row.append(q*scale)
        outs.append(torch.cat(row))
    return torch.stack(outs)

r = ref(x, gs)
trailing = y[:, gs*((n_cols+gs-1)//gs):]  # empty here; use last partial group indices
start = (n_cols // gs) * gs  # 64
print("trailing slice start:", start, "n_cols:", n_cols)
tr = y[:, start:]
tr_ref = r[:, start:]
tr_x = x[:, start:]
out = {
  "shape": list(x.shape),
  "group_size": gs,
  "trailing_cols_zero": bool((tr == 0).all().item()),
  "trailing_input_nonzero_count": int((tr_x != 0).sum().item()),
  "max_abs_err_trailing_vs_ref": float((tr - tr_ref).abs().max().item()),
  "full_output_max_abs_err": float((y - r).abs().max().item()),
  "trailing_out_sample": tr[0, :8].tolist(),
  "trailing_ref_sample": tr_ref[0, :8].tolist(),
  "trailing_x_sample": tr_x[0, :8].tolist(),
}
print(json.dumps(out))
