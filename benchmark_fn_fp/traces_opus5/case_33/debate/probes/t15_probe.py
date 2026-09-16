
import subprocess, sys, json, os

script = r'''
import importlib.util, json, numpy as np, torch, traceback
spec = importlib.util.spec_from_file_location("kk", "/root/cases/case_33/kernel.py")
kk = importlib.util.module_from_spec(spec); spec.loader.exec_module(kk)
dev='cuda'; rng=np.random.default_rng(1)
M,N,K = 64,64,40
gs = 16
G = -(-K//gs)   # 3
q = rng.integers(0,16,size=(K,N)).astype(np.uint32)
rows = -(-K//8)  # 5
pk = np.zeros((rows,N),dtype=np.uint32)
for k in range(K): pk[k//8] |= (q[k]&15) << (4*(k%8))

# ALL group rows identical -> group misassignment (c1) cannot contribute any error
zqrow = np.full(N,7,dtype=np.uint32)
zprow = np.zeros(N//8,dtype=np.uint32)
for n in range(N): zprow[n//8] |= (zqrow[n]&15) << (4*(n%8))
zp = np.ascontiguousarray(np.stack([zprow]*G))
sc = np.full((G,N),2.5,dtype=np.float32)

a_np = rng.standard_normal((M,K)).astype(np.float32)
gidx = np.arange(K)//gs
deq = (q.astype(np.float64) - (zqrow[None,:].astype(np.float64)+1.0)) * 2.5
ref = a_np.astype(np.float64) @ deq
scale_ref = float(np.abs(ref).max())

res = {"shape":[M,N,K],"group_size":gs,"group_rows":G,"packed_rows_alloc":rows,
       "loop_tiles":-(-K//16),"tiles_cover_k_upto":16*(-(-K//16)),
       "ref_max_abs":scale_ref,"all_group_rows_identical":True}

# exact-size allocations: a is (M,40), b_packed is (5,N), g_idx is (40,)
a = torch.from_numpy(a_np).to(dev)
b = torch.from_numpy(np.ascontiguousarray(pk.view(np.int32))).to(dev)
scales = torch.from_numpy(sc).to(dev)
zw = torch.from_numpy(np.ascontiguousarray(zp.view(np.int32))).to(dev)
try:
    c = kk.gptq_matmul(a,b,scales,zw,gs,4); torch.cuda.synchronize()
    cn = c.double().cpu().numpy()
    res["k40_cuda_error"] = None
    res["k40_max_abs_err_vs_ref"] = float(np.abs(cn-ref).max())
    res["k40_rel_err_vs_ref"] = float(np.abs(cn-ref).max()/scale_ref)
    res["k40_frac_entries_err_gt_1pct"] = float((np.abs(cn-ref)>0.01*scale_ref).mean())
    res["k40_nonfinite_count"] = int((~np.isfinite(cn)).sum())
except Exception as e:
    res["k40_cuda_error"] = repr(e)[:400]
    print(json.dumps(res)); raise SystemExit(0)

# control: zero-pad K to 48 so the same 3 tiles read only legal, zero-contributing data
Kp = 48
a_pad = np.zeros((M,Kp),dtype=np.float32); a_pad[:,:K] = a_np
pkp = np.zeros((Kp//8,N),dtype=np.uint32); pkp[:rows] = pk
Gp = Kp//gs   # 3, clamp inactive
scp = np.full((Gp,N),2.5,dtype=np.float32)
zpp = np.ascontiguousarray(np.stack([zprow]*Gp))
try:
    cp = kk.gptq_matmul(torch.from_numpy(a_pad).to(dev),
                        torch.from_numpy(np.ascontiguousarray(pkp.view(np.int32))).to(dev),
                        torch.from_numpy(scp).to(dev),
                        torch.from_numpy(np.ascontiguousarray(zpp.view(np.int32))).to(dev),
                        gs,4)
    torch.cuda.synchronize()
    cpn = cp.double().cpu().numpy()
    res["padded_k48_cuda_error"]=None
    res["padded_k48_max_abs_err_vs_ref"]=float(np.abs(cpn-ref).max())
    res["padded_k48_rel_err_vs_ref"]=float(np.abs(cpn-ref).max()/scale_ref)
    res["k40_vs_padded_max_abs_diff"]=float(np.abs(cn-cpn).max())
except Exception as e:
    res["padded_k48_cuda_error"]=repr(e)[:400]
print(json.dumps(res))
'''
open('/tmp/c2_probe.py','w').write(script)
env = dict(os.environ); env["CUDA_LAUNCH_BLOCKING"] = "1"
p = subprocess.run([sys.executable,'/tmp/c2_probe.py'],capture_output=True,text=True,timeout=420,env=env)
inner = None
for line in reversed([l for l in p.stdout.splitlines() if l.strip()]):
    try:
        inner = json.loads(line); break
    except Exception:
        continue
print(json.dumps({"returncode":p.returncode,"inner":inner,
                  "stderr_tail":p.stderr[-1500:],"stdout_tail":p.stdout[-500:]}))
