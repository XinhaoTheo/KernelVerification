import json, torch
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_24/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
a = torch.randn(4, 300, device="cuda", dtype=torch.float32)
b = torch.randn(4, 300, device="cuda", dtype=torch.float32)
# give b a different row stride than a: transpose a wider tensor
b_wide = torch.randn(300, 4, device="cuda", dtype=torch.float32).t()  # shape (4,300), stride (1,300)
b = b_wide.contiguous().t().contiguous()  # placeholder; use actual non-matching stride below
b = torch.randn(4, 600, device="cuda", dtype=torch.float32)[:, ::2]  # shape (4,300), stride(0)=600
print("a.stride:", a.stride(), "b.stride:", b.stride())

out = k.cosine_similarity(a, b)
ref = torch.nn.functional.cosine_similarity(a, b, dim=1)
diff = (out - ref).abs()
res = {"a_stride0": a.stride(0), "b_stride0": b.stride(0),
       "max_abs_err": float(diff.max()), "mean_abs_err": float(diff.mean()),
       "out": out.tolist(), "ref": ref.tolist(),
       "metric": "row-wise absolute error vs torch F.cosine_similarity",
       "note": "b has row stride 600 != a.stride(0)=300; kernel uses a.stride(0) for both"}
print(json.dumps(res))