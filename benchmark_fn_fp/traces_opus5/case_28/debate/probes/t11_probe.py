
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_28/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev="cuda"; C=256
def rel(y,x): return ((y-x).norm(dim=1)/x.norm(dim=1))

# row1: constant magnitude 0.3 with signs (1 distinct |value|)
r1 = torch.full((1,C), 0.3, device=dev); r1[0,1::2] = -0.3
# row2: two distinct magnitudes + zeros (sparse post-ReLU style)
r2 = torch.zeros(1,C, device=dev); r2[0,:10]=0.2; r2[0,10:20]=0.4
x = torch.cat([r1,r2], dim=0)
y = k.quant_dequant(x)
r = rel(y,x)
out = {
 "shape": list(x.shape),
 "rel_l2_constant_row": round(r[0].item(),5),
 "rel_l2_sparse_2mag_row": round(r[1].item(),5),
 "tolerance": 0.05,
 "constant_row_out_all_zero": bool(y[0].abs().max().item()==0.0),
 "sparse_row_out_all_zero": bool(y[1].abs().max().item()==0.0),
 "constant_row_max_abs_out": round(y[0].abs().max().item(),6),
 "sparse_row_max_abs_out": round(y[1].abs().max().item(),6),
 "n_distinct_abs_row0": int(torch.unique(x[0].abs()).numel()),
 "n_distinct_abs_row1": int(torch.unique(x[1].abs()).numel()),
}
print(json.dumps(out))
