
import torch, json, importlib.util, math
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(1)
dev='cuda'

def ref32(X, eps):
    x = X.float()
    ms = (x*x).sum(dim=1, keepdim=True)/x.shape[1]
    return x*torch.rsqrt(ms+eps)

res={}
def case(name, X, eps):
    Y = k.rms_norm_forward(X, eps)
    R = ref32(X, eps)
    d = (Y-R).abs()
    fin = torch.isfinite(R)
    res[name]=dict(shape=list(X.shape), eps=eps,
        max_abs=float(d[fin].max()) if fin.any() else None,
        max_rel=float((d[fin]/(R[fin].abs()+1e-38)).max()) if fin.any() else None,
        ref_nonfinite=int((~fin).sum()), out_nonfinite=int((~torch.isfinite(Y)).sum()),
        y_absmax=float(Y[torch.isfinite(Y)].abs().max()) if torch.isfinite(Y).any() else None)

eps=1e-6
# extreme magnitudes
for s,name in [(1e-25,'tiny_1e-25'),(1e-38,'subnormal_1e-38'),(1e-45,'subnormal_min'),(1e18,'huge_1e18'),(1e20,'huge_1e20')]:
    X = torch.randn(4,128,device=dev)*s
    case(name, X, eps)

# mixed rows in one tensor: huge, tiny, normal, zero
X = torch.randn(4,128,device=dev)
X[0]*=1e20; X[1]*=1e-25; X[3]*=0.0
case('mixed_magnitudes', X, eps)

# large n_cols
for n in [2048, 4096, 8192, 16384]:
    try:
        case(f'ncols_{n}', torch.randn(2,n,device=dev), eps)
    except Exception as e:
        res[f'ncols_{n}']={'error': repr(e)[:200]}

# bf16 roundtrip of the *same* tensor: compare kernel(bf16) to fp32-ref of bf16 values
Xf = torch.randn(64,512,device=dev)*0.001
Xb = Xf.bfloat16()
Yb = k.rms_norm_forward(Xb, eps)
Rb = ref32(Xb.float(), eps)
res['bf16_small_scale']=dict(max_abs=float((Yb-Rb).abs().max()),
                             max_rel=float(((Yb-Rb).abs()/(Rb.abs()+1e-38)).max()))

# near-zero row where mean_square underflows: check rstd path exactness
Xz = torch.full((3,64), 1e-24, device=dev)
Yz = k.rms_norm_forward(Xz, eps)
Rz = ref32(Xz, eps)
res['underflow_row']=dict(y0=float(Yz[0,0]), ref0=float(Rz[0,0]),
                          match=bool(torch.equal(Yz,Rz)))

# eps interplay: values comparable to sqrt(eps)
Xe = torch.full((2,32), 1e-3, device=dev)
Ye = k.rms_norm_forward(Xe, 1e-6); Re = ref32(Xe, 1e-6)
res['eps_comparable']=dict(y=float(Ye[0,0]), ref=float(Re[0,0]),
                           analytic=1e-3/math.sqrt(1e-6+1e-6))
print("JSONSTART")
print(json.dumps(res))
