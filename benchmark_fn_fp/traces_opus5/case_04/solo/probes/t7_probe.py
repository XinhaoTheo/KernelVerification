
import torch, json, importlib.util, math
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(0)
dev='cuda'

def ref(X, eps):
    x = X.float()
    ms = (x*x).sum(dim=1, keepdim=True)/x.shape[1]
    return x*torch.rsqrt(ms+eps)

res={}
def case(name, X, eps):
    Y = k.rms_norm_forward(X, eps)
    R = ref(X, eps)
    ae = (Y-R).abs()
    rel = ae/(R.abs()+1e-30)
    finite = torch.isfinite(R)
    res[name]=dict(shape=list(X.shape), in_dtype=str(X.dtype), out_dtype=str(Y.dtype),
                   max_abs=float(ae[finite].max()) if finite.any() else None,
                   max_rel=float(rel[finite].max()) if finite.any() else None,
                   ref_absmax=float(R[finite].abs().max()) if finite.any() else None,
                   out_nonfinite=int((~torch.isfinite(Y)).sum()))

eps=1e-6
case("fp32_basic", torch.randn(8,64,device=dev), eps)
case("fp32_nonpow2", torch.randn(37,1000,device=dev)*3, eps)
case("fp32_manyrows", torch.randn(4096,127,device=dev), eps)
case("fp32_1col", torch.randn(5,1,device=dev), eps)
case("fp32_1row_bigcols", torch.randn(1,4096,device=dev), eps)

# near-zero rows
X = torch.randn(6,64,device=dev)
X[0]*=0.0
X[1]*=1e-8
X[2]*=1e-20
X[3]*=1e-30
X[4]*=1e-3
case("nearzero", X, eps)

# bf16 round trip
Xb = (torch.randn(16,256,device=dev)).bfloat16()
case("bf16", Xb, eps)
case("bf16_nearzero", (X.bfloat16()), eps)

# fp32 vs bf16-roundtrip consistency: kernel on bf16 vs kernel on bf16->fp32
Xb32 = Xb.float()
Y1 = k.rms_norm_forward(Xb, eps).float(); Y2 = k.rms_norm_forward(Xb32, eps)
res["bf16_vs_fp32roundtrip_maxabs"]=float((Y1-Y2).abs().max())

# varying eps
for e in [0.0, 1e-12, 1e-5, 1.0]:
    case(f"eps_{e}", torch.randn(4,300,device=dev)*0.01, e)

# rstd sanity: check row independence via padded columns leaking
Xp = torch.zeros(3,100,device=dev); Xp[0,:]=1.0; Xp[1,:]=2.0; Xp[2,:]=0.0
Yp = k.rms_norm_forward(Xp, eps)
res["explicit_rows"]=dict(row0=float(Yp[0,0]), row1=float(Yp[1,0]), row2_max=float(Yp[2].abs().max()),
                          expect0=1/math.sqrt(1+eps), expect1=2/math.sqrt(4+eps))
print(json.dumps(res, indent=1))
