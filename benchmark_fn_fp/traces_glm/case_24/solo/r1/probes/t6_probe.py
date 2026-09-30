
import torch, json, sys
sys.path.insert(0, "/root/cases/case_24")
from kernel import cosine_similarity

dev = "cuda"
torch.manual_seed(0)

def ref(a, b, eps=1e-8):
    a32 = a.double(); b32 = b.double()
    dot = (a32*b32).sum(-1)
    denom = torch.sqrt((a32*a32).sum(-1)) * torch.sqrt((b32*b32).sum(-1))
    return (dot / torch.clamp(denom, min=eps)).float()

results = []
for (M, N) in [(4, 64), (128, 1000), (37, 513), (1, 1), (256, 4096)]:
    a = torch.randn(M, N, device=dev, dtype=torch.float32)
    b = torch.randn(M, N, device=dev, dtype=torch.float32)
    out = cosine_similarity(a, b)
    r = ref(a, b)
    results.append(dict(shape=(M,N), max_abs=(out-r).abs().max().item(), in_bounds=bool(((out.abs()<=1.0+1e-5).all().item())))

# zero-norm rows: a row all zeros, b row all zeros, both zeros
a = torch.randn(8, 300, device=dev); b = torch.randn(8, 300, device=dev)
a[2] = 0.0; b[4] = 0.0
a[6] = 0.0; b[6] = 0.0
out = cosine_similarity(a, b)
r = ref(a, b)
results.append(dict(shape="zero-norm rows", max_abs=(out-r).abs().max().item(), finite=bool(torch.isfinite(out).all().item()), ref_vals=r[2].item(), out_val=out[2].item()))

# tiny-norm rows (denominator below eps)
a = torch.full((4, 128), 1e-5, device=dev); b = torch.full((4,128), -1e-5, device=dev)
out = cosine_similarity(a, b); r = ref(a, b)
results.append(dict(shape="tiny-norm", max_abs=(out-r).abs().max().item(), finite=bool(torch.isfinite(out).all().item())))

# parallel/orthogonal/antiparallel exact cases
a = torch.eye(3, device=dev); b = torch.eye(3, device=dev)
out = cosine_similarity(a, b)
results.append(dict(shape="identity", out=out.tolist()))
a = torch.tensor([[1.,2.,3.],[1.,0.,0.],[0.,1.,0.]], device=dev)
b = torch.tensor([[-1.,-2.,-3.],[0.,1.,0.],[0.,1.,0.]], device=dev)
out = cosine_similarity(a, b); r = ref(a,b)
results.append(dict(shape="aligned", out=out.tolist(), ref=r.tolist(), max_abs=(out-r).abs().max().item()))

print(json.dumps(results, indent=1))
overall = max(x.get("max_abs", 0.0) for x in results if isinstance(x.get("max_abs"), float))
print(json.dumps({"overall_max_abs": overall, "pass": overall < 1e-5}))
